"""Hand-engineered features for hybrid models and the RF baseline.

These 18 features capture pulse-rate-band power, signal shape, and variability —
exactly the kinds of summaries a cardiologist looks at when reading a PPG strip.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew

FS = 50      # sampling frequency in Hz (one PPG sample every 20 ms)
NPERSEG = 128  # segment length for Welch's power-spectral-density estimate

FEATURE_NAMES = [
    "mean", "std", "var", "skew", "kurtosis", "peak_to_peak",
    "sum_abs_diff", "mean_crossings",
    "psd_sum", "dominant_freq",
    "psd_band_0.5-1.0Hz", "psd_band_1.0-1.5Hz", "psd_band_1.5-2.0Hz",
    "psd_band_2.0-3.0Hz", "psd_band_3.0-4.0Hz", "psd_band_4.0-5.0Hz",
    "spectral_entropy", "cardiac_band_power_ratio",
]


def extract_features(X: np.ndarray) -> np.ndarray:
    """Compute 18-D feature vector per window.

    Args:
        X: (N, 512) array of PPG windows (raw or LP-filtered, NOT z-scored —
           z-scoring kills the absolute-amplitude features).

    Returns:
        (N, 18) float32 feature matrix, NaN/inf replaced with 0.
    """
    feats = []
    # Time-domain
    feats.append(X.mean(axis=1))
    feats.append(X.std(axis=1))
    feats.append(X.var(axis=1))
    feats.append(skew(X, axis=1))
    feats.append(kurtosis(X, axis=1))
    feats.append(X.max(axis=1) - X.min(axis=1))               # peak-to-peak
    feats.append(np.sum(np.abs(np.diff(X, axis=1)), axis=1))  # sum |dX|
    feats.append(np.sum(np.diff(np.sign(X - X.mean(axis=1, keepdims=True)),
                                 axis=1) != 0, axis=1))       # mean crossings

    # Frequency-domain via Welch PSD
    f, psd = welch(X, fs=FS, nperseg=NPERSEG, axis=1)
    mask = (f >= 0.5) & (f <= 5.0)
    f_sub, psd_sub = f[mask], psd[:, mask]
    feats.append(psd_sub.sum(axis=1))                          # psd_sum
    feats.append(f_sub[psd_sub.argmax(axis=1)])                # dominant frequency

    # PSD band powers in HR-relevant ranges
    for lo, hi in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                   (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]:
        m = (f >= lo) & (f < hi)
        feats.append(psd[:, m].sum(axis=1))

    # Spectral entropy (how "spread out" the spectrum is)
    psd_norm = psd_sub / (psd_sub.sum(axis=1, keepdims=True) + 1e-12)
    feats.append(entropy(psd_norm.T, base=2))

    # Cardiac-band (0.8–2.5 Hz, ~48–150 bpm) power ratio
    m_card = (f >= 0.8) & (f < 2.5)
    feats.append(psd[:, m_card].sum(axis=1) / (psd_sub.sum(axis=1) + 1e-12))

    F = np.stack(feats, axis=1).astype(np.float32)
    F[np.isnan(F) | np.isinf(F)] = 0.0
    return F


def standardize(F_train: np.ndarray, *others: np.ndarray):
    """Z-score features using train-set statistics only (avoid leakage)."""
    mu = F_train.mean(axis=0, keepdims=True)
    sd = F_train.std(axis=0, keepdims=True) + 1e-8
    return [(arr - mu) / sd for arr in (F_train, *others)]
