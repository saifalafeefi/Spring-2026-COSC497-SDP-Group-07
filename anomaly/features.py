"""hand-crafted features for the statistical baseline.

stress shows up in PPG as HR ↑ and HRV ↓, so the feature set leans on pulse rate
and pulse-interval variability, plus a few shape/spectral descriptors. computed
on the wrist BVP window; amplitude-invariant where it matters (we z-score first)
so sensor gain doesn't leak in.

returns a fixed-length vector; NaNs (e.g. too few detectable pulses) are left in
and handled by the detector at fit time.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

FEATURE_NAMES = [
    "hr_bpm",          # pulse rate from median peak interval
    "ibi_sdnn",        # std of peak-to-peak intervals (HRV ↓ under stress)
    "ibi_rmssd",       # rms of successive interval diffs
    "n_peaks_per_s",   # detected pulses per second
    "bp_std",          # std of the band-passed pulse
    "bp_ptp",          # peak-to-peak of the band-passed pulse
    "dom_freq",        # dominant frequency in the pulse band
    "pulse_band_ratio",# power in 0.7–3 Hz vs total
    "spec_entropy",    # spectral entropy (flat spectrum = noisy)
    "mean_abs_diff",   # roughness of the raw window
]
N_FEATURES = len(FEATURE_NAMES)


def _bandpass(x, fs, lo=0.7, hi=3.0):
    sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def extract_features(window: np.ndarray, fs: int = 64) -> np.ndarray:
    """one BVP window → feature vector (len N_FEATURES)."""
    x = np.asarray(window, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-8)        # z-score: kill sensor gain
    f = np.full(N_FEATURES, np.nan)

    try:
        bp = _bandpass(x, fs)
    except Exception:
        bp = x
    f[4] = bp.std()
    f[5] = bp.max() - bp.min()
    f[9] = np.mean(np.abs(np.diff(x)))

    # pulse peaks → HR + HRV
    prom = bp.std() * 0.3 if bp.std() > 0 else None
    peaks, _ = find_peaks(bp, distance=int(fs * 0.4), prominence=prom)
    if len(peaks) >= 3:
        ibi = np.diff(peaks) / fs                 # inter-beat intervals (s)
        f[0] = 60.0 / np.median(ibi)
        f[1] = np.std(ibi)
        f[2] = np.sqrt(np.mean(np.diff(ibi) ** 2))
    f[3] = len(peaks) / (len(x) / fs)

    # spectrum
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    psd = np.abs(np.fft.rfft(x)) ** 2
    tot = psd.sum() + 1e-12
    band = (freqs >= 0.7) & (freqs <= 3.0)
    f[6] = freqs[band][psd[band].argmax()] if band.any() and psd[band].sum() > 0 else np.nan
    f[7] = psd[band].sum() / tot
    p = psd / tot
    f[8] = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return f


def extract_batch(windows: np.ndarray, fs: int = 64) -> np.ndarray:
    """(n_win × win_len) → (n_win × N_FEATURES)."""
    return np.array([extract_features(w, fs) for w in windows], dtype=np.float32)
