"""live inference wrapper for the saved autoencoder — used by the dashboard.

loads anomaly/saved/ae.keras + scorer.npz and turns one BVP window into:
  • score  — raw reconstruction error
  • level  — 0..1 display value (calm ≈ 0, clearly anomalous ≈ 1)
  • flag   — bool, score past the saved threshold
"""
from __future__ import annotations

import os
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")


class LiveAnomalyDetector:
    def __init__(self, save_dir: str | None = None):
        import tensorflow as tf
        save_dir = save_dir or SAVE_DIR
        self.model = tf.keras.models.load_model(
            os.path.join(save_dir, "ae.keras"), compile=False)
        z = np.load(os.path.join(save_dir, "scorer.npz"))
        self.threshold = float(z["threshold"])
        self.win_len = int(z["win_len"])
        self.ref_lo = float(z["ref_lo"])
        self.ref_hi = float(z["ref_hi"])

    def score(self, window: np.ndarray) -> float:
        x = np.asarray(window, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)
        recon = self.model.predict(x[None, :, None], verbose=0)[0, :, 0]
        return float(np.mean((recon - x) ** 2))

    def level(self, score: float) -> float:
        return float(np.clip((score - self.ref_lo) / (self.ref_hi - self.ref_lo + 1e-9),
                             0.0, 1.0))

    def flag(self, score: float) -> bool:
        return score >= self.threshold
