"""per-user calibration: zero-shot vs device-calibrated.

the subject-variance in the LOSO results says a detector built on OTHER people
transfers unevenly. the realistic fix is to calibrate to the user with a short
slice of their OWN calm at setup. this experiment quantifies that lift on WESAD
— the "zero-shot vs device-calibrated" delta O6 asks for, demonstrated on public
data before our own device data exists.

three regimes, all scored on the SAME held-out test set (apples-to-apples):
  zero-shot   : fit on every OTHER subject's calm
  calibrated  : fit on the held-out subject's OWN calm (a leading time slice)
  hybrid      : fit on both (population prior + personal) — the deployable option

the subject's calm is split time-respecting: the first `--calib-frac` is the
"onboarding" calibration data, the rest (+ all their stress) is the test set.

    python3 -m anomaly.calibrate
    python3 -m anomaly.calibrate --calib-frac 0.5
"""
from __future__ import annotations

import argparse
import numpy as np

from .run import load_all, NORMAL, POSITIVE
from .wesad import SUBJECTS
from .features import extract_batch
from .baseline import MahalanobisDetector
from .metrics import summarize, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="fraction of the subject's calm used to calibrate")
    args = ap.parse_args()

    data = load_all(SUBJECTS, args.win, args.step)
    print("\nextracting features once per subject…")
    feats = {}
    for s in SUBJECTS:
        X, c = data[s]
        feats[s] = (extract_batch(X[c == 1]), extract_batch(X[c == 2]))

    rows = {"zero-shot": [], "calibrated": [], "hybrid": []}
    print(f"\nLOSO  (calib-frac {args.calib_frac:g})")
    print(f"{'subj':5s} {'zero-shot':>10s} {'calibrated':>11s} {'hybrid':>8s}   (PR-AUC)")
    for test in SUBJECTS:
        Fcalm, Fstress = feats[test]
        k = max(5, int(len(Fcalm) * args.calib_frac))
        Fcal, Ftest_calm = Fcalm[:k], Fcalm[k:]
        Fte = np.vstack([Ftest_calm, Fstress])
        y = np.r_[np.zeros(len(Ftest_calm)), np.ones(len(Fstress))].astype(int)
        Fother = np.vstack([feats[s][0] for s in SUBJECTS if s != test])

        scores = {
            "zero-shot":  MahalanobisDetector().fit_features(Fother).score_features(Fte),
            "calibrated": MahalanobisDetector().fit_features(Fcal).score_features(Fte),
            "hybrid":     MahalanobisDetector().fit_features(np.vstack([Fother, Fcal])).score_features(Fte),
        }
        line = [test]
        for r in ("zero-shot", "calibrated", "hybrid"):
            m = summarize(y, scores[r]); m["subject"] = test
            rows[r].append(m); line.append(m["pr_auc"])
        print(f"{line[0]:5s} {line[1]:10.3f} {line[2]:11.3f} {line[3]:8.3f}")

    print("\n=== mean ± std across subjects ===")
    agg = {r: aggregate(rows[r]) for r in rows}
    print(f"{'regime':12s} {'PR-AUC':>16s} {'recall@90spec':>18s}")
    for r in ("zero-shot", "calibrated", "hybrid"):
        pr = agg[r]["pr_auc"]; rc = agg[r]["recall@90spec"]
        print(f"{r:12s} {pr[0]:.3f} ± {pr[1]:.3f}      {rc[0]:.3f} ± {rc[1]:.3f}")
    d_pr = agg["calibrated"]["pr_auc"][0] - agg["zero-shot"]["pr_auc"][0]
    d_pr_h = agg["hybrid"]["pr_auc"][0] - agg["zero-shot"]["pr_auc"][0]
    print(f"\ncalibration delta (PR-AUC): calibrated {d_pr:+.3f} · hybrid {d_pr_h:+.3f} vs zero-shot")


if __name__ == "__main__":
    main()
