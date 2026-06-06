"""WESAD loader for the one-class anomaly pipeline.

first brick of the eval harness. it pulls the **wrist BVP @ 64 Hz** channel —
the consumer-grade PPG that stands in for our cheap sensor — aligns the WESAD
label track onto the BVP timeline, and slices clean fixed-length windows.

we deliberately ignore the 700 Hz chest signals (ECG/RespiBAN): using them would
be cheating on signal quality, which is the whole point of the project.

WESAD labels (700 Hz track): 1=baseline · 2=stress(TSST) · 3=amusement ·
4=meditation · 0/5/6/7 = transient/prep → discarded.

quickstart
----------
    from anomaly.wesad import load_subject, make_windows, to_binary

    s = load_subject("S2")
    X, cond = make_windows(s["bvp"], s["labels"], win_sec=60, step_sec=5)
    y = to_binary(cond)           # 0 = normal(baseline), 1 = stress, -1 = drop

cli
---
    python3 anomaly/wesad.py            # summary across all subjects
    python3 anomaly/wesad.py S2 --win 60 --step 5
"""
from __future__ import annotations

import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WESAD_DIR = os.path.join(REPO_ROOT, "WESAD")

FS = 64                      # wrist BVP sample rate (Hz)
FS_ACC = 32                 # wrist accelerometer sample rate (Hz)

# the 15 released subjects (no S1, no S12)
SUBJECTS = [f"S{i}" for i in range(2, 18) if i not in (1, 12)]

# usable protocol conditions on the WESAD label track
COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
USABLE = set(COND)          # everything else (0, 5, 6, 7) is transient → drop


def load_subject(subject: str, wesad_dir: str | None = None,
                 with_acc: bool = False) -> dict:
    """load one subject's wrist BVP and its label track (aligned to 64 Hz).

    returns a dict: subject, fs, bvp (float32, n), labels (int8, n), and — if
    with_acc — acc (float32, m×3) at FS_ACC. the big chest arrays are read then
    dropped, so peak memory is one pickle, not the whole dataset.
    """
    wesad_dir = wesad_dir or WESAD_DIR
    path = os.path.join(wesad_dir, subject, f"{subject}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. WESAD lives at {wesad_dir} (gitignored, ~17 GB).")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")   # WESAD pickles are Python-2 made

    bvp = np.asarray(d["signal"]["wrist"]["BVP"]).reshape(-1).astype(np.float32)
    label700 = np.asarray(d["label"]).reshape(-1)
    acc = (np.asarray(d["signal"]["wrist"]["ACC"]).astype(np.float32)
           if with_acc else None)
    del d   # release the 700 Hz chest arrays asap

    # align the 700 Hz label track onto the 64 Hz BVP timeline. both streams
    # share a start time in the synchronized pickle, so we map by length ratio
    # (== 700/64) rather than assuming an exact integer factor.
    n = len(bvp)
    idx = (np.arange(n) * (len(label700) / n)).astype(np.int64)
    np.clip(idx, 0, len(label700) - 1, out=idx)
    labels = label700[idx].astype(np.int8)

    out = {"subject": subject, "fs": FS, "bvp": bvp, "labels": labels}
    if with_acc:
        out["acc"] = acc
        out["fs_acc"] = FS_ACC
    return out


def make_windows(x: np.ndarray, labels: np.ndarray, win_sec: float = 60.0,
                 step_sec: float = 5.0, fs: int = FS,
                 pure: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """slide fixed windows over a 1-D signal and label each by its condition.

    pure=True (default) keeps only windows that sit entirely inside ONE usable
    condition — windows straddling a transition or touching a transient label
    are dropped, so every returned window has a clean ground-truth condition.

    returns X (n_win × win_len, float32) and cond (n_win, int8 in {1,2,3,4}).
    """
    win = int(round(win_sec * fs))
    step = max(1, int(round(step_sec * fs)))
    Xs, ys = [], []
    for s in range(0, len(x) - win + 1, step):
        lw = labels[s:s + win]
        if pure:
            uniq = np.unique(lw)
            if uniq.size != 1 or int(uniq[0]) not in USABLE:
                continue
            cond = int(uniq[0])
        else:
            vals, counts = np.unique(lw, return_counts=True)
            cond = int(vals[counts.argmax()])
            if cond not in USABLE:
                continue
        Xs.append(x[s:s + win])
        ys.append(cond)
    if not Xs:
        return (np.empty((0, win), np.float32), np.empty((0,), np.int8))
    return np.asarray(Xs, np.float32), np.asarray(ys, np.int8)


def to_binary(cond: np.ndarray, positive=(2,), normal=(1,)) -> np.ndarray:
    """map condition labels to a binary task: 0 = normal, 1 = positive, -1 = drop.

    default = the standard WESAD stress task (baseline vs stress). amusement and
    meditation fall to -1 so the caller can decide whether to fold them into
    'normal' for one-class training or leave them out.
    """
    out = np.full(cond.shape, -1, dtype=np.int8)
    out[np.isin(cond, normal)] = 0
    out[np.isin(cond, positive)] = 1
    return out


def iter_subjects(subjects=None, **kw):
    """yield load_subject(...) for each subject — one pickle in memory at a time."""
    for s in (subjects or SUBJECTS):
        yield load_subject(s, **kw)


def _summary(subjects, win_sec, step_sec):
    print(f"WESAD @ {WESAD_DIR}")
    print(f"windows: {win_sec:g}s / {step_sec:g}s step  ({int(win_sec*FS)} samples @ {FS} Hz)\n")
    hdr = f"{'subj':5s} {'minutes':>7s} | " + " ".join(f"{c:>10s}" for c in COND.values())
    print(hdr); print("-" * len(hdr))
    tot = {c: 0 for c in COND}
    for subj in subjects:
        s = load_subject(subj)
        X, cond = make_windows(s["bvp"], s["labels"], win_sec, step_sec)
        per = {c: int((cond == c).sum()) for c in COND}
        for c in COND:
            tot[c] += per[c]
        mins = len(s["bvp"]) / FS / 60
        print(f"{subj:5s} {mins:7.1f} | " + " ".join(f"{per[c]:>10d}" for c in COND))
    print("-" * len(hdr))
    print(f"{'TOTAL':5s} {'':>7s} | " + " ".join(f"{tot[c]:>10d}" for c in COND))
    print(f"\nbinary stress task: {tot[1]} normal(baseline) vs {tot[2]} stress windows")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect WESAD wrist-BVP windows")
    ap.add_argument("subjects", nargs="*", default=SUBJECTS,
                    help="subject ids (default: all 15)")
    ap.add_argument("--win", type=float, default=60.0, help="window seconds")
    ap.add_argument("--step", type=float, default=5.0, help="step seconds")
    args = ap.parse_args()
    _summary(args.subjects, args.win, args.step)
