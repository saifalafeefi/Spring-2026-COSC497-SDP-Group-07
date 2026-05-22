"""Classifier API for inference on PPG windows.

Quickstart
----------
    from inference_lib import Classifier
    import numpy as np

    clf = Classifier()
    ppg_window = np.array([...])              # shape (512,) raw PPG samples @ 50 Hz
    result = clf.classify(ppg_window)
    print(result.label, result.confidence)    # 'cardiac' | 'non_cardiac' | 'occlusion', 0.0–1.0

For embedded deployment (int8 TFLite):
    clf = Classifier.from_tflite()            # loads model_int8.tflite from the same run
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

CLASSES = ["cardiac", "non_cardiac", "occlusion"]
EXPECTED_WINDOW_LEN = 512  # 10.24 s @ 50 Hz

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
FINAL_PRESET = "phase_a"


@dataclass
class ClassificationResult:
    label: str                       # "cardiac" | "non_cardiac" | "occlusion"
    confidence: float                # 0.0 to 1.0
    probabilities: dict[str, float]  # per-class probabilities

    def __repr__(self):
        probs = ", ".join(f"{k}={v:.3f}" for k, v in self.probabilities.items())
        return (f"ClassificationResult(label={self.label!r}, "
                f"confidence={self.confidence:.3f}, [{probs}])")


def _final_run() -> str:
    candidates = sorted(d for d in os.listdir(RUNS_DIR) if d.endswith(f"_{FINAL_PRESET}"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {RUNS_DIR}. "
            f"Run `python3 baselines/train.py --preset {FINAL_PRESET}` first.")
    return os.path.join(RUNS_DIR, candidates[-1])


def _zscore(window: np.ndarray) -> np.ndarray:
    """Per-window mean-0, std-1 normalization. Matches training preprocessing."""
    mu = window.mean()
    sd = window.std() + 1e-8
    return ((window - mu) / sd).astype(np.float32)


class Classifier:
    """Loads the final trained PPG classifier and exposes a single
    ``classify(ppg_window)`` method."""

    def __init__(self, run_dir: str | None = None):
        """Load the latest trained final model (Keras format).

        Args:
            run_dir: Optional explicit run directory. Defaults to the most
                recent ``runs/*_phase_a`` folder.
        """
        import tensorflow as tf
        self.run_dir = run_dir or _final_run()
        self._keras = tf.keras.models.load_model(
            os.path.join(self.run_dir, "model.keras"), compile=False)
        self._interp = None
        self.needs_features = True  # final model is the hybrid CNN + features

    @classmethod
    def from_tflite(cls, run_dir: str | None = None) -> "Classifier":
        """Load the int8 TFLite version (for embedded/ESP32 deployment)."""
        import tensorflow as tf
        run = run_dir or _final_run()
        path = os.path.join(run, "model_int8.tflite")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run "
                f"`python3 baselines/quantize.py --preset {FINAL_PRESET}` first.")
        obj = cls.__new__(cls)
        obj.run_dir = run
        obj._keras = None
        obj._interp = tf.lite.Interpreter(model_path=path)
        obj._interp.allocate_tensors()
        obj.needs_features = True
        return obj

    # ---------- public API ----------

    def classify(self, ppg_window: np.ndarray) -> ClassificationResult:
        """Predict the class of a single PPG window.

        Args:
            ppg_window: 1-D numpy array of 512 raw PPG samples (any scale;
                z-scored internally). Sampled at 50 Hz.

        Returns:
            ClassificationResult with label, confidence, and per-class probabilities.
        """
        ppg_window = np.asarray(ppg_window).squeeze()
        if ppg_window.ndim != 1 or ppg_window.shape[0] != EXPECTED_WINDOW_LEN:
            raise ValueError(
                f"Expected 1-D array of {EXPECTED_WINDOW_LEN} samples, "
                f"got shape {ppg_window.shape}.")

        raw = ppg_window.astype(np.float32)
        sig = _zscore(raw)[None, :, None]
        feat = self._compute_features(raw)

        probs = self._predict(sig, feat)
        idx = int(np.argmax(probs))
        return ClassificationResult(
            label=CLASSES[idx],
            confidence=float(probs[idx]),
            probabilities={c: float(p) for c, p in zip(CLASSES, probs)},
        )

    # ---------- internals ----------

    def _predict(self, sig: np.ndarray, feat: np.ndarray) -> np.ndarray:
        if self._keras is not None:
            out = self._keras.predict([sig, feat[None]], verbose=0)
            return out[0]
        inputs = self._interp.get_input_details()
        sig_in = next(d for d in inputs if len(d["shape"]) == 3)
        feat_in = next(d for d in inputs if len(d["shape"]) == 2)
        self._interp.set_tensor(sig_in["index"], self._maybe_quantize(sig, sig_in))
        self._interp.set_tensor(feat_in["index"], self._maybe_quantize(feat[None], feat_in))
        self._interp.invoke()
        out_det = self._interp.get_output_details()[0]
        out = self._interp.get_tensor(out_det["index"])[0]
        if out_det["dtype"] == np.int8:
            scale, zp = out_det["quantization"]
            out = (out.astype(np.float32) - zp) * scale
        return out

    @staticmethod
    def _maybe_quantize(arr: np.ndarray, det: dict) -> np.ndarray:
        if det["dtype"] == np.int8:
            scale, zp = det["quantization"]
            return np.clip(np.round(arr / scale + zp), -128, 127).astype(np.int8)
        return arr.astype(np.float32)

    def _compute_features(self, raw_signal: np.ndarray) -> np.ndarray:
        from features import extract_features
        F = extract_features(raw_signal[None])
        mu, sd = self._feature_stats()
        return ((F[0] - mu) / sd).astype(np.float32)

    def _feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_cached_stats"):
            return self._cached_stats
        from data import load_dataset, participant_split
        from features import extract_features
        with open(os.path.join(self.run_dir, "config.json")) as fp:
            cfg = json.load(fp)
        ds_raw = load_dataset(placement=cfg["placement"], zscore=False)
        tr, _, _ = participant_split(ds_raw, seed=cfg["seed"])
        F_tr = extract_features(ds_raw.X[tr])
        mu = F_tr.mean(axis=0)
        sd = F_tr.std(axis=0) + 1e-8
        self._cached_stats = (mu, sd)
        return self._cached_stats
