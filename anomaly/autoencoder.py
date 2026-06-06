"""1D-conv autoencoder — the O1 detector.

train it to reconstruct NORMAL (baseline) wrist-BVP windows only. at test time,
windows it reconstructs poorly (high MSE) are flagged anomalous — the model has
never learned to represent stress, so stress should reconstruct worse.

train heavy / deploy light: this trains offline; only the small encoder + a tiny
score head would ship to the device later (the project's stretch path).

TensorFlow is imported lazily so the statistical baseline path never pays for it.
"""
from __future__ import annotations

import numpy as np


def _zscore(X: np.ndarray) -> np.ndarray:
    """per-window mean-0 / std-1 (kills sensor gain; model learns shape)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def build_ae(win_len: int, n_blocks: int = 4, base: int = 16):
    """fully-convolutional AE. win_len must be divisible by 2**n_blocks."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    if win_len % (2 ** n_blocks) != 0:
        raise ValueError(f"win_len {win_len} must be divisible by {2**n_blocks}")

    inp = layers.Input(shape=(win_len, 1))
    x = inp
    for i in range(n_blocks):                       # encoder: halve length each block
        x = layers.Conv1D(base * (2 ** i), 7, strides=2, padding="same",
                          activation="relu")(x)
    for i in reversed(range(n_blocks)):             # decoder: double back up
        x = layers.Conv1DTranspose(base * (2 ** i), 7, strides=2, padding="same",
                                   activation="relu")(x)
    out = layers.Conv1D(1, 7, padding="same")(x)    # reconstruct, length == win_len
    m = Model(inp, out, name="bvp_ae")
    m.compile(optimizer="adam", loss="mse")
    return m


class AEDetector:
    """reconstruction-error one-class detector."""

    def __init__(self, win_len: int, n_blocks: int = 4, base: int = 16,
                 epochs: int = 30, batch: int = 64, seed: int = 0):
        self.win_len = win_len
        self.n_blocks = n_blocks
        self.base = base
        self.epochs = epochs
        self.batch = batch
        self.seed = seed
        self.model = None

    def fit(self, normal_windows: np.ndarray, verbose: int = 0):
        import tensorflow as tf
        tf.random.set_seed(self.seed)
        X = _zscore(normal_windows)[..., None]
        self.model = build_ae(self.win_len, self.n_blocks, self.base)
        self.model.fit(X, X, epochs=self.epochs, batch_size=self.batch,
                       validation_split=0.1, shuffle=True, verbose=verbose,
                       callbacks=self._early_stop())
        return self

    def _early_stop(self):
        import tensorflow as tf
        return [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)]

    def score(self, windows: np.ndarray) -> np.ndarray:
        """per-window reconstruction MSE (higher = more anomalous)."""
        X = _zscore(windows)[..., None]
        recon = self.model.predict(X, batch_size=256, verbose=0)
        return np.mean((recon - X) ** 2, axis=(1, 2))
