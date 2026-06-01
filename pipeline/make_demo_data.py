"""Generate a multi-channel curated PPG + IMU demo file.

Picks 3 high-confidence windows per class from the trained model's perspective
and stitches them into a single time-series CSV.

Channels written (all 50 Hz, one row per sample):
  - sample_idx
  - ir            real PPG IR-LED value from the UBC dataset
  - red           plausible RED-LED value (synthesized — UBC LP set only
                  has the IR channel, but the deployed MAX30102 has both)
  - accel_x/y/z   plausible 3-axis accelerometer trace (synthesized — UBC
                  per-window labels don't span the IMU recordings cleanly)
  - true_label    cardiac | non_cardiac | occlusion

For the deployed device this synthesis goes away — the MAX30102 + MPU6050
will provide all 7 channels directly.

Run once after the dataset is present:
    python3 pipeline/make_demo_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines"))

from data import CLASSES, load_dataset                # noqa: E402
from inference_lib import Classifier                  # noqa: E402

OUT_PATH = os.path.join(HERE, "demo_data.csv")
SEED = 7
N_CANDIDATES = 60
WINDOWS_PER_CLASS = 3
FS = 50

# Synthetic fall injected into the last segment so the FallDetector fires
# during the demo. Removed when real accel comes from the MPU6050.
INJECT_FALL = True
FALL_AT_S = 88.0   # seconds into the stream — late in the occlusion segment
FALL_FREEFALL_MS = 180
FALL_IMPACT_MS = 80
FALL_IMPACT_G = 4.2
FALL_STILLNESS_MS = 1500

# Per-class synthesis profiles (RED LED + IMU)
#
# RED is synthesised by scaling the IR's DC and AC components INDEPENDENTLY:
#   RED = red_dc_scale·mean(IR) + red_ac_scale·(IR − mean(IR)) + noise
# The R-ratio = AC_red/DC_red ÷ AC_IR/DC_IR = red_ac_scale / red_dc_scale.
# SpO2 ≈ 110 − 25·R, so:
#   R=0.52 → SpO2≈97%   R=0.60 → SpO2≈95%   R=0.80 → SpO2≈90%
PROFILES = {
    "cardiac": {
        "red_dc_scale": 0.85,
        "red_ac_scale": 0.44,            # R≈0.52 → SpO2≈97 %
        "red_noise_std": 15,             # noise « AC signal
        "accel_g": (0.02, 0.02, 0.99),
        "accel_noise_std": 0.015,
        "accel_breath_amp": 0.01,
    },
    "non_cardiac": {
        "red_dc_scale": 0.0,             # off-body — RED dominated by noise
        "red_ac_scale": 0.0,
        "red_noise_std": 800,
        "accel_g": (0.01, 0.02, 0.99),
        "accel_noise_std": 0.05,
        "accel_breath_amp": 0.0,
    },
    "occlusion": {
        "red_dc_scale": 0.85,
        "red_ac_scale": 0.66,            # R≈0.78 → SpO2≈90 % (mild desat)
        "red_noise_std": 5,              # occlusion AC is small; keep noise tiny
        "accel_g": (0.03, 0.03, 0.98),
        "accel_noise_std": 0.018,
        "accel_breath_amp": 0.008,
    },
}


def _pick_top_examples(ds, class_idx: int, clf: Classifier, rng, k: int) -> list[int]:
    pool = np.where(ds.y == class_idx)[0]
    candidates = rng.choice(pool, size=min(N_CANDIDATES, len(pool)), replace=False)
    target = CLASSES[class_idx]
    scored = []
    for i in candidates:
        res = clf.classify(ds.X[i])
        scored.append((res.probabilities[target], int(i)))
    scored.sort(reverse=True)
    chosen = [i for _, i in scored[:k]]
    confs = [c for c, _ in scored[:k]]
    print(f"  {target}: confidences {[f'{c:.2f}' for c in confs]}")
    return chosen


def _synth_red(ir: np.ndarray, profile: dict, rng) -> np.ndarray:
    """Generate a plausible RED-LED trace from the IR signal.

    Independent DC and AC scaling so the R-ratio (and therefore SpO2) is
    physically plausible — see PROFILES docstring above."""
    if profile["red_dc_scale"] == 0.0:
        baseline = float(np.mean(ir))
        return baseline * 0.05 + rng.normal(0, profile["red_noise_std"], size=ir.shape)
    dc_ir = float(np.mean(ir))
    ac_ir = ir - dc_ir
    red = (profile["red_dc_scale"] * dc_ir
           + profile["red_ac_scale"] * ac_ir
           + rng.normal(0, profile["red_noise_std"], size=ir.shape))
    return red


def _synth_imu(n_samples: int, profile: dict, rng) -> np.ndarray:
    """Return (n_samples, 3) accelerometer trace in 'g' units."""
    t = np.arange(n_samples) / FS
    breath = profile["accel_breath_amp"] * np.sin(2 * np.pi * 0.25 * t)  # ~15 breaths/min
    noise = rng.normal(0, profile["accel_noise_std"], size=(n_samples, 3))
    gx, gy, gz = profile["accel_g"]
    accel = np.stack([
        np.full(n_samples, gx),
        np.full(n_samples, gy) + breath,    # most respiration shows up on one axis
        np.full(n_samples, gz),
    ], axis=1) + noise
    return accel


def main():
    rng = np.random.default_rng(SEED)
    print("Loading dataset (raw, no z-score) ...")
    ds = load_dataset(placement="all", zscore=False)
    print("Loading classifier to select clean demo samples ...")
    clf = Classifier()

    all_rows = []
    sample_counter = 0
    for class_idx, cls in enumerate(CLASSES):
        profile = PROFILES[cls]
        for win_idx in _pick_top_examples(ds, class_idx, clf, rng, WINDOWS_PER_CLASS):
            ir = ds.X[win_idx].astype(np.float64)
            red = _synth_red(ir, profile, rng)
            accel = _synth_imu(len(ir), profile, rng)
            for i in range(len(ir)):
                all_rows.append({
                    "sample_idx": sample_counter,
                    "ir": float(ir[i]),
                    "red": float(red[i]),
                    "accel_x": float(accel[i, 0]),
                    "accel_y": float(accel[i, 1]),
                    "accel_z": float(accel[i, 2]),
                    "true_label": cls,
                })
                sample_counter += 1

    df = pd.DataFrame(all_rows)

    if INJECT_FALL:
        df = _inject_fall(df, rng)
        print(f"\nInjected synthetic fall starting at t = {FALL_AT_S:.1f} s")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(df):,} samples, {len(df)/FS:.1f} s @ {FS} Hz")
    for cls, n in df.true_label.value_counts().items():
        print(f"    {cls:12s}: {n:4d} samples ({n/FS:.1f} s)")


def _inject_fall(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Overwrite the accel_* columns around FALL_AT_S with a fall signature.
    Free-fall ~0.1 g → impact spike ~4 g → stillness ~1 g with tiny noise."""
    start = int(FALL_AT_S * FS)
    n_ff = int(FALL_FREEFALL_MS * FS / 1000)
    n_imp = int(FALL_IMPACT_MS * FS / 1000)
    n_still = int(FALL_STILLNESS_MS * FS / 1000)
    end = start + n_ff + n_imp + n_still
    if end > len(df):
        # extend the dataframe by repeating last row's other channels
        n_extra = end - len(df)
        last_row = df.iloc[-1].copy()
        extras = pd.DataFrame([last_row] * n_extra)
        extras["sample_idx"] = np.arange(len(df), len(df) + n_extra)
        df = pd.concat([df, extras], ignore_index=True)

    # column indices for fast iloc assignment
    cx = df.columns.get_loc("accel_x")
    cy = df.columns.get_loc("accel_y")
    cz = df.columns.get_loc("accel_z")

    # phase 1: free-fall — accel near zero
    i0, i1 = start, start + n_ff
    df.iloc[i0:i1, cx] = rng.normal(0, 0.05, n_ff)
    df.iloc[i0:i1, cy] = rng.normal(0, 0.05, n_ff)
    df.iloc[i0:i1, cz] = rng.normal(0, 0.05, n_ff)

    # phase 2: impact — sharp Z spike
    i0, i1 = i1, i1 + n_imp
    df.iloc[i0:i1, cx] = rng.normal(0, 0.10, n_imp)
    df.iloc[i0:i1, cy] = rng.normal(0, 0.10, n_imp)
    df.iloc[i0:i1, cz] = rng.normal(FALL_IMPACT_G, 0.15, n_imp)

    # phase 3: stillness — back to ~1 g on Z, very low noise
    i0, i1 = i1, i1 + n_still
    df.iloc[i0:i1, cx] = rng.normal(0.0, 0.005, n_still)
    df.iloc[i0:i1, cy] = rng.normal(0.0, 0.005, n_still)
    df.iloc[i0:i1, cz] = rng.normal(1.0, 0.005, n_still)
    return df


if __name__ == "__main__":
    main()
