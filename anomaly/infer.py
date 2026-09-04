"""live inference wrapper for the deployed autoencoder — used by the dashboard.

runs the **int8 TFLite** model (anomaly/saved/ae_int8.tflite) — the exact artifact
that ships to the ESP32-S3 — so the dashboard shows what the device actually does.
the scorer (`scorer.npz` from anomaly.compress, calibrated on the int8 score
distribution over WESAD; or `scorer_device.npz` from anomaly.device_calibrate,
re-derived on our own sensor) turns one BVP window into:
  • score  — raw reconstruction error
  • level  — 0..1 display value (calm ≈ 0, clearly anomalous ≈ 1)
  • flag   — bool, score past the saved threshold
"""
from __future__ import annotations

import os
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
WESAD_SCORER = "scorer.npz"          # written by anomaly.compress, WESAD wrist BVP
DEVICE_SCORER = "scorer_device.npz"  # written by anomaly.device_calibrate, our own rig


def resolve_scorer(mode: str = "replay", save_dir: str | None = None) -> str:
    """which scorer file a given source should flag against.

    the device scorer only exists once `anomaly.device_calibrate` has been run.
    until then the device falls back to the WESAD one — which is exactly the
    documented caveat: the waveform is real, the flag is not yet meaningful,
    because a wrist-BVP threshold is being applied to a fingertip sensor.
    """
    if mode == "device":
        if os.path.exists(os.path.join(save_dir or SAVE_DIR, DEVICE_SCORER)):
            return DEVICE_SCORER
    return WESAD_SCORER


class LiveAnomalyDetector:
    def __init__(self, save_dir: str | None = None, scorer: str | None = None):
        import tensorflow as tf
        save_dir = save_dir or SAVE_DIR
        self.scorer_name = scorer or WESAD_SCORER
        z = np.load(os.path.join(save_dir, self.scorer_name))
        # what the thresholds were derived on: "wesad" (public data) or "device"
        self.calibrated_on = str(z["source"]) if "source" in z else "wesad"
        self.threshold = float(z["threshold"])
        self.win_len = int(z["win_len"])
        self.ref_lo = float(z["ref_lo"])
        self.ref_hi = float(z["ref_hi"])

        self.interp = tf.lite.Interpreter(
            model_path=os.path.join(save_dir, "ae_int8.tflite"))
        in0 = self.interp.get_input_details()[0]
        self.interp.resize_tensor_input(in0["index"], [1, self.win_len, 1])
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def score(self, window: np.ndarray) -> float:
        x = np.asarray(window, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)
        xin = x[None, :, None]
        if self.inp["dtype"] == np.int8:                  # quantize the input
            s, zp = self.inp["quantization"]
            xin = np.clip(np.round(xin / s + zp), -128, 127)
        self.interp.set_tensor(self.inp["index"], xin.astype(self.inp["dtype"]))
        self.interp.invoke()
        r = self.interp.get_tensor(self.out["index"])[0, :, 0].astype(np.float32)
        if self.out["dtype"] == np.int8:                  # dequantize the output
            s, zp = self.out["quantization"]
            r = (r - zp) * s
        return float(np.mean((r - x) ** 2))

    def level(self, score: float) -> float:
        return float(np.clip((score - self.ref_lo) / (self.ref_hi - self.ref_lo + 1e-9),
                             0.0, 1.0))

    def score_for_level(self, level: float) -> float:
        """inverse of level(): the raw MSE score that maps to a 0–1 display level.
        used to turn a user-chosen sensitivity (a level threshold) into the MSE
        threshold the flag actually compares against."""
        return self.ref_lo + level * (self.ref_hi - self.ref_lo)

    def flag(self, score: float) -> bool:
        return score >= self.threshold
