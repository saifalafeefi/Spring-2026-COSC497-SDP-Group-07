"""train one deployable autoencoder on all subjects' calm windows and save it.

unlike run.py (which does leave-one-subject-out for evaluation), this trains a
single model on everyone's baseline data and saves it for the live dashboard:

    anomaly/saved/ae.keras       the trained autoencoder
    anomaly/saved/scorer.npz     threshold + display-scaling constants

    python3 -m anomaly.export --epochs 40
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from .run import load_all, NORMAL, POSITIVE
from .wesad import SUBJECTS
from .autoencoder import AEDetector

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--spec", type=float, default=0.90,
                    help="flag threshold = this specificity on calm windows")
    ap.add_argument("--denoise", type=float, default=0.0, metavar="SIGMA",
                    help="train a denoising AE (e.g. 0.15); 0 = off. validate with "
                         "`anomaly.run --model ae --denoise SIGMA` before deploying.")
    ap.add_argument("--bottleneck", type=int, default=0, metavar="DIM",
                    help="AE latent width (e.g. 256); 0 = original. validate with "
                         "`anomaly.run --model ae --bottleneck DIM` before deploying.")
    ap.add_argument("--blocks", type=int, default=4, metavar="N",
                    help="AE encoder depth; pair with --ch-cap for an ESP32-sized bottleneck.")
    ap.add_argument("--ch-cap", type=int, default=0, metavar="C", dest="ch_cap",
                    help="cap encoder channels (e.g. 32); shrinks the bottleneck Dense.")
    args = ap.parse_args()

    win_len = int(round(args.win * 64))
    data = load_all(SUBJECTS, args.win, args.step)
    Xn = np.concatenate([data[s][0][np.isin(data[s][1], list(NORMAL))] for s in data])
    Xs = np.concatenate([data[s][0][np.isin(data[s][1], list(POSITIVE))] for s in data])
    print(f"\ntraining AE on {len(Xn)} calm windows ({args.epochs} epochs)…")

    det = AEDetector(win_len=win_len, epochs=args.epochs, noise=args.denoise,
                     bottleneck=args.bottleneck, n_blocks=args.blocks,
                     ch_cap=args.ch_cap).fit(Xn, verbose=0)

    mse_n = det.score(Xn)
    mse_s = det.score(Xs)
    thr = float(np.quantile(mse_n, args.spec))
    recall = float(np.mean(mse_s >= thr))
    print(f"calm score: median {np.median(mse_n):.4f}  threshold@{args.spec:.0%} {thr:.4f}")
    print(f"stress score median {np.median(mse_s):.4f}  → recall at threshold {recall:.2f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    det.model.save(os.path.join(SAVE_DIR, "ae.keras"))
    np.savez(os.path.join(SAVE_DIR, "scorer.npz"),
             threshold=thr, win_len=win_len,
             ref_lo=float(np.median(mse_n)),          # display: 0% level
             ref_hi=float(np.quantile(mse_n, 0.99)))  # display: ~100% level
    print(f"saved → {SAVE_DIR}/ae.keras + scorer.npz")


if __name__ == "__main__":
    main()
