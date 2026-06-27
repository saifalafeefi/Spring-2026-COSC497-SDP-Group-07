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


def _corrupt(Xc: np.ndarray, noise: float, fs: int, rng) -> np.ndarray:
    """corrupt clean z-scored windows the way a cheap wrist PPG would: amplitude
    drift, slow baseline wander, and sensor jitter — then re-z-score so the input
    distribution matches what the model sees at inference time.

    used only for the denoising path: the model learns to map a NOISY pulse back
    to its clean shape, so it keys on pulse morphology (which stress changes) and
    is less fooled by motion/contact noise on our real hardware. Xc, return: (n, L).
    """
    n, L = Xc.shape
    x = Xc.copy()
    t = np.linspace(0, L / fs, L)[None, :]                 # seconds
    # random per-window amplitude (loose contact / gain drift)
    x = x * rng.uniform(0.7, 1.3, size=(n, 1))
    # slow baseline wander (respiration / motion drift, 0.05–0.3 Hz)
    f = rng.uniform(0.05, 0.3, size=(n, 1))
    x = x + rng.uniform(0.1, 0.5, size=(n, 1)) * np.sin(2 * np.pi * f * t
                                                        + rng.uniform(0, 2 * np.pi, size=(n, 1)))
    # white sensor jitter, scaled by `noise`
    x = x + rng.normal(0.0, noise, size=x.shape)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


def build_ae(win_len: int, n_blocks: int = 4, base: int = 16, bottleneck: int = 0,
             ch_cap: int = 0):
    """1D-conv AE. win_len must be divisible by 2**n_blocks.

    bottleneck > 0 inserts a real compressed latent (a Dense layer of that width)
    between encoder and decoder. WITHOUT it the conv "latent" is 240x128 = 30,720
    values for a 3,840-sample window — 8x LARGER than the input, i.e. over-complete:
    the AE can near-copy anything (including stress), so anomalies reconstruct too
    well and barely separate. forcing the signal through e.g. 256 units makes the
    model learn only the normal-pulse manifold, so stress reconstructs poorly and
    stands out. bottleneck=0 reproduces the original (validated) architecture.

    SIZE / ESP32 NOTE: the Dense bottleneck's cost = enc_len*enc_ch*bottleneck. with
    the default 4 blocks that flatten is 240*128 = 30,720 → ~16M params (~16 MB int8),
    fine for a Pi but FAR too big for an ESP32. to keep the bottleneck benefit in a
    tiny model, downsample harder (more `n_blocks`) and cap channels (`ch_cap`) so the
    flatten — and thus the Dense — is small. e.g. n_blocks=8, ch_cap=32, bottleneck=256
    → flatten 15*32=480 → ~0.3M params (~0.3 MB int8), ESP32-deployable.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    if win_len % (2 ** n_blocks) != 0:
        raise ValueError(f"win_len {win_len} must be divisible by {2**n_blocks}")

    def ch(i):                                      # channel width, optionally capped
        c = base * (2 ** i)
        return min(c, ch_cap) if ch_cap and ch_cap > 0 else c

    inp = layers.Input(shape=(win_len, 1))
    x = inp
    for i in range(n_blocks):                       # encoder: halve length each block
        x = layers.Conv1D(ch(i), 7, strides=2, padding="same", activation="relu")(x)

    if bottleneck and bottleneck > 0:               # true compressed latent
        enc_len = win_len // (2 ** n_blocks)
        enc_ch = ch(n_blocks - 1)
        x = layers.Flatten()(x)
        x = layers.Dense(bottleneck, activation="relu", name="latent")(x)
        x = layers.Dense(enc_len * enc_ch, activation="relu")(x)
        x = layers.Reshape((enc_len, enc_ch))(x)

    for i in reversed(range(n_blocks)):             # decoder: double back up
        x = layers.Conv1DTranspose(ch(i), 7, strides=2, padding="same",
                                   activation="relu")(x)
    out = layers.Conv1D(1, 7, padding="same")(x)    # reconstruct, length == win_len
    m = Model(inp, out, name="bvp_ae")
    m.compile(optimizer="adam", loss="mse")
    return m


class AEDetector:
    """reconstruction-error one-class detector."""

    def __init__(self, win_len: int, n_blocks: int = 4, base: int = 16,
                 epochs: int = 30, batch: int = 64, seed: int = 0,
                 noise: float = 0.0, aug_copies: int = 2, fs: int = 64,
                 bottleneck: int = 0, ch_cap: int = 0):
        self.win_len = win_len
        self.n_blocks = n_blocks
        self.base = base
        self.ch_cap = ch_cap        # cap encoder channels → keeps the Dense bottleneck small
        self.epochs = epochs
        self.batch = batch
        self.seed = seed
        self.noise = noise          # >0 → denoising AE (corrupt input, clean target)
        self.aug_copies = aug_copies
        self.fs = fs
        self.bottleneck = bottleneck  # >0 → real compressed latent (fixes over-completeness)
        self.model = None

    def fit(self, normal_windows: np.ndarray, verbose: int = 0):
        import tensorflow as tf
        tf.random.set_seed(self.seed)
        Xc = _zscore(normal_windows)                       # clean, z-scored (n, L)
        self.model = build_ae(self.win_len, self.n_blocks, self.base,
                              self.bottleneck, self.ch_cap)

        if self.noise and self.noise > 0:
            # denoising: noisy copies → clean target, plus one clean→clean anchor.
            rng = np.random.default_rng(self.seed)
            ins = [Xc] + [_corrupt(Xc, self.noise, self.fs, rng)
                          for _ in range(self.aug_copies)]
            Xin = np.concatenate(ins, axis=0)
            Xtg = np.tile(Xc, (1 + self.aug_copies, 1))
            perm = rng.permutation(len(Xin))               # shuffle before val split
            Xin, Xtg = Xin[perm], Xtg[perm]
        else:
            Xin = Xtg = Xc

        self.model.fit(Xin[..., None], Xtg[..., None], epochs=self.epochs,
                       batch_size=self.batch, validation_split=0.1, shuffle=True,
                       verbose=verbose, callbacks=self._early_stop())
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
