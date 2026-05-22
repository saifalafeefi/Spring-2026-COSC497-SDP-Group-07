"""End-to-end inference demo.

Loads the trained model, runs it on a sample of windows from each class,
and reports prediction accuracy. If this prints reasonable results, the
inference pipeline is working.

Run from repo root:  python3 baselines/inference_demo.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from data import CLASSES, load_dataset
from inference_lib import Classifier


def _pick_per_class(ds, rng, n_per_class: int = 10):
    picks = []
    for cls_idx in range(len(CLASSES)):
        idx_pool = np.where(ds.y == cls_idx)[0]
        chosen = rng.choice(idx_pool, size=n_per_class, replace=False)
        for i in chosen:
            picks.append((ds.X[i], CLASSES[cls_idx]))
    return picks


def main():
    rng = np.random.default_rng(123)
    print("Loading PPG dataset (raw signal, no z-score) ...")
    ds = load_dataset(placement="all", zscore=False)
    samples = _pick_per_class(ds, rng, n_per_class=10)
    print(f"Picked {len(samples)} samples (10 per class).\n")

    clf = Classifier()
    print(f"Loaded model from: {clf.run_dir}\n")

    correct = 0
    per_class = defaultdict(lambda: [0, 0])
    for raw_signal, true_label in samples:
        res = clf.classify(raw_signal)
        ok = res.label == true_label
        correct += int(ok)
        per_class[true_label][0] += int(ok)
        per_class[true_label][1] += 1

    print(f"Overall: {correct}/{len(samples)} correct "
          f"({correct/len(samples)*100:.0f}%)")
    for cls in CLASSES:
        ok, n = per_class[cls]
        print(f"  {cls:12s}: {ok}/{n}")


if __name__ == "__main__":
    main()
