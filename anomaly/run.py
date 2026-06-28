"""leave-one-subject-out evaluation harness (O1).

ties the bricks together: load WESAD → window wrist BVP → for each held-out
subject, fit a one-class detector on every OTHER subject's NORMAL windows, score
the held-out subject's normal + stress windows, and report PR-AUC + recall@90%
specificity. aggregates as mean ± std across subjects — the honest number.

    python3 -m anomaly.run --model baseline                  # all 15 subjects
    python3 -m anomaly.run --model ae --epochs 30
    python3 -m anomaly.run --model baseline --max-subjects 3  # quick check

(WESAD is ~17 GB and gitignored — this loads one subject pickle at a time.)
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from .wesad import load_subject, make_windows, SUBJECTS, WESAD_DIR
from .splits import leave_one_subject_out
from .metrics import summarize, aggregate, episode_metrics

NORMAL = {1}        # baseline
POSITIVE = {2}      # stress (TSST)

# cached windowed data lives inside WESAD/ (already gitignored) so the 13 GB of
# pickles are read once per (win, step), then reused across baseline/ae/ssl runs.
CACHE_DIR = os.path.join(WESAD_DIR, "_harness_cache")


def load_all(subjects, win_sec, step_sec):
    """subject -> (windows, cond), built from pickles once and cached to .npz."""
    path = os.path.join(CACHE_DIR, f"win{win_sec:g}_step{step_sec:g}.npz")
    cache = {}
    if os.path.exists(path):
        npz = np.load(path)
        cache = {k: npz[k] for k in npz.files}

    data, new = {}, False
    for s in subjects:
        kx, kc = f"{s}_X", f"{s}_c"
        if kx in cache and kc in cache:
            data[s] = (cache[kx], cache[kc])
            src = "cached"
        else:
            d = load_subject(s)
            X, cond = make_windows(d["bvp"], d["labels"], win_sec, step_sec)
            keep = np.isin(cond, list(NORMAL | POSITIVE))
            data[s] = (X[keep], cond[keep])
            cache[kx], cache[kc] = data[s]
            new, src = True, "read"
        n_norm = int(np.isin(data[s][1], list(NORMAL)).sum())
        n_str = int(np.isin(data[s][1], list(POSITIVE)).sum())
        print(f"  {src:6s} {s}: {n_norm} normal / {n_str} stress windows")

    if new:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, **cache)
    return data


def make_detector(args, win_len):
    if args.model == "baseline":
        from .baseline import MahalanobisDetector
        return MahalanobisDetector(fs=64)
    if args.model == "ae":
        from .autoencoder import AEDetector
        return AEDetector(win_len=win_len, epochs=args.epochs,
                          noise=args.denoise, bottleneck=args.bottleneck,
                          n_blocks=args.blocks, ch_cap=args.ch_cap)
    from .ssl import SSLDetector
    return SSLDetector(win_len=win_len, epochs=args.epochs)


def main():
    ap = argparse.ArgumentParser(description="LOSO one-class stress detection on WESAD")
    ap.add_argument("--model", choices=["baseline", "ae", "ssl"], default="baseline")
    ap.add_argument("--win", type=float, default=60.0, help="window seconds")
    ap.add_argument("--step", type=float, default=5.0, help="step seconds")
    ap.add_argument("--epochs", type=int, default=30, help="AE epochs")
    ap.add_argument("--denoise", type=float, default=0.0, metavar="SIGMA",
                    help="denoising AE: train on noise-corrupted windows (e.g. 0.15). "
                         "0 = off (the validated default). only affects --model ae.")
    ap.add_argument("--bottleneck", type=int, default=0, metavar="DIM",
                    help="AE latent width (e.g. 256). >0 adds a real compressed bottleneck "
                         "— fixes the over-complete AE and sharpens anomaly separation. "
                         "0 = original architecture. only affects --model ae.")
    ap.add_argument("--blocks", type=int, default=4, metavar="N",
                    help="AE encoder depth (stride-2 blocks). more = harder downsampling "
                         "before the bottleneck = smaller Dense = ESP32-sized model. "
                         "win must be divisible by 2**N (3840 → up to 8). default 4.")
    ap.add_argument("--ch-cap", type=int, default=0, metavar="C", dest="ch_cap",
                    help="cap encoder channels at C (e.g. 32). shrinks the bottleneck Dense. "
                         "0 = uncapped (original). pair with --blocks for an ESP32 model.")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="use only the first N subjects (quick checks)")
    ap.add_argument("--episode-k", type=int, default=3, dest="episode_k", metavar="K",
                    help="episode-level metric: a stress episode counts as detected only "
                         "with >=K consecutive flagged windows (pre-committed, default 3).")
    args = ap.parse_args()

    subjects = SUBJECTS[: args.max_subjects] if args.max_subjects else SUBJECTS
    win_len = int(round(args.win * 64))
    print(f"model={args.model}  win={args.win:g}s ({win_len} samp)  step={args.step:g}s  "
          f"subjects={len(subjects)}\nloading…")
    data = load_all(subjects, args.win, args.step)

    rows = []
    print("\nLOSO:")
    for train, test in leave_one_subject_out(subjects):
        Xtr = np.concatenate([data[s][0][np.isin(data[s][1], list(NORMAL))]
                              for s in train])
        Xte, cte = data[test]
        yte = np.isin(cte, list(POSITIVE)).astype(int)
        det = make_detector(args, win_len).fit(Xtr)
        sc = det.score(Xte)
        m = summarize(yte, sc)
        m.update(episode_metrics(sc, yte, step_sec=args.step, k=args.episode_k))
        m["subject"] = test
        rows.append(m)
        print(f"  {test:5s}  PR-AUC={m['pr_auc']:.3f}  "
              f"recall@90spec={m['recall@90spec']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
              f"| ep={m['ep_recall']:.0f} fa/h={m['fa_per_hr']:.2f} lat={m['latency_s']:.0f}s  "
              f"(+{m['n_pos']}/-{m['n_neg']})")

    agg = aggregate(rows)
    print("\n=== mean ± std across subjects ===")
    for k, (mean, sd) in agg.items():
        print(f"  {k:14s} {mean:.3f} ± {sd:.3f}")
    print(f"\n  episode-level (K={args.episode_k} sustained windows @ 90% spec): "
          f"ep_recall = fraction of stress episodes caught · fa_per_hr = false alarms / "
          f"hour of calm · latency_s = median time-to-flag")


if __name__ == "__main__":
    main()
