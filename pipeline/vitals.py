"""Lightweight derived-metric helpers.

Pure functions over numpy arrays — no UI, no model dependencies.
Used by the dashboard to surface human-readable vitals alongside the
classifier output.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def estimate_heart_rate(ir_window: np.ndarray, fs: int = 50) -> float | None:
    """Estimate heart rate in BPM from a PPG IR window.

    Band-passes the signal to the plausible HR range (0.7–3 Hz ≈ 42–180 BPM),
    finds peaks, and converts the median peak-to-peak interval to BPM.

    Returns None when the signal is too short or too noisy.
    """
    if len(ir_window) < fs * 4:
        return None
    try:
        sos = butter(N=2, Wn=[0.7, 3.0], btype="band", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, ir_window)
    except Exception:
        return None
    min_distance = int(fs * 0.4)
    prom = filtered.std() * 0.4 if filtered.std() > 0 else None
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=prom)
    if len(peaks) < 2:
        return None
    intervals_s = np.diff(peaks) / fs
    median_interval = float(np.median(intervals_s))
    if median_interval <= 0:
        return None
    bpm = 60.0 / median_interval
    if not (30 <= bpm <= 200):
        return None
    return float(bpm)


def estimate_spo2(ir_window: np.ndarray, red_window: np.ndarray) -> float | None:
    """Estimate SpO₂ (%) from IR and RED channels using the standard R-ratio.

    R = (AC_red / DC_red) / (AC_IR / DC_IR)
    SpO₂ ≈ 110 - 25·R    (empirical formula used in consumer pulse oximeters)

    Returns None if either channel is too flat / contains no pulsatile signal.
    """
    if len(ir_window) < 30 or len(red_window) < 30:
        return None
    ac_ir = float(np.std(ir_window))
    dc_ir = float(np.mean(ir_window))
    ac_red = float(np.std(red_window))
    dc_red = float(np.mean(red_window))
    if dc_ir <= 0 or dc_red <= 0:
        return None
    if ac_ir < 30 or ac_red < 10:   # essentially flat — sensor probably off-body
        return None
    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    if not (0.3 <= R <= 2.0):       # implausible — almost always bad signal
        return None
    spo2 = 110.0 - 25.0 * R
    return float(max(70.0, min(100.0, spo2)))


def signal_quality_score(ir_window: np.ndarray) -> float:
    """0-1 signal-quality estimate from amplitude consistency."""
    if len(ir_window) < 50:
        return 0.0
    sd = float(ir_window.std())
    ptp = float(ir_window.max() - ir_window.min())
    if ptp == 0:
        return 0.0
    ratio = sd / ptp
    return max(0.0, min(1.0, (ratio - 0.05) / 0.25))


def motion_intensity(accel_xyz: np.ndarray, fs: int = 50) -> float:
    """Return the standard deviation of accel magnitude (in 'g') over the window.

    accel_xyz : (N, 3) array — last N samples of (ax, ay, az).
    A value near 0 means stationary; > 0.1 means visible motion.
    """
    if accel_xyz.shape[0] < 10:
        return 0.0
    mag = np.linalg.norm(accel_xyz, axis=1)
    return float(np.std(mag))


# Note: the proper 4-phase fall detector lives in `fall_detector.py`. This
# module only handles stateless single-window vitals (HR, SpO2, etc.).
