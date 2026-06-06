"""self-supervised encoder (O2) — contrastive pretraining.

idea: teach a small 1D-conv encoder the *shape* of normal pulse signals without
any labels, then do anomaly detection in the learned embedding space.

how the self-supervision works (SimCLR / NT-Xent, contrastive style):
  • take a batch of windows, make two lightly distorted copies of each (jitter +
    scaling) — two "views" of the same heartbeat snippet
  • train the encoder so the two views of the SAME window land close together in
    embedding space, and different windows land far apart
no labels needed — the supervision comes from "these two came from the same
window." after pretraining we keep the encoder, embed the normal windows, and
fit a Gaussian to them; test windows far from that Gaussian are flagged.

same fit/score interface as baseline.py and autoencoder.py, so run.py treats it
identically. TensorFlow is imported lazily.
"""
from __future__ import annotations

import numpy as np


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def build_encoder(win_len: int, embed_dim: int = 64, base: int = 16,
                  n_blocks: int = 4):
    """1D-conv encoder → fixed-size embedding (global pooled, any length ok)."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(win_len, 1))
    x = inp
    for i in range(n_blocks):
        x = layers.Conv1D(base * (2 ** i), 7, strides=2, padding="same",
                          activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(embed_dim)(x)
    return Model(inp, out, name="bvp_encoder")


def _projection_head(embed_dim, proj_dim=32):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(embed_dim,))
    x = layers.Dense(embed_dim, activation="relu")(inp)
    out = layers.Dense(proj_dim)(x)
    return Model(inp, out, name="proj_head")


class SSLDetector:
    """contrastive-pretrained encoder + Gaussian scorer in embedding space."""

    def __init__(self, win_len: int, embed_dim: int = 64, epochs: int = 50,
                 batch: int = 128, temp: float = 0.5, noise: float = 0.1,
                 reg: float = 1e-3, seed: int = 0):
        self.win_len = win_len
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch = batch
        self.temp = temp
        self.noise = noise
        self.reg = reg
        self.seed = seed
        self.encoder = None

    # ---- contrastive pieces ----

    def _augment(self, x):
        import tensorflow as tf
        x = x + tf.random.normal(tf.shape(x)) * self.noise          # jitter
        scale = tf.random.uniform((tf.shape(x)[0], 1, 1), 0.8, 1.2)  # scaling
        return x * scale

    def _nt_xent(self, z):
        """contrastive loss: pull the 2 views of each window together."""
        import tensorflow as tf
        z = tf.math.l2_normalize(z, axis=1)
        n = tf.shape(z)[0]
        b = n // 2
        sim = tf.matmul(z, z, transpose_b=True) / self.temp
        sim = sim - tf.eye(n) * 1e9                       # mask self-similarity
        targets = tf.concat([tf.range(b, n), tf.range(0, b)], axis=0)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(targets, sim))

    # ---- fit / score ----

    def fit(self, normal_windows: np.ndarray, verbose: int = 0):
        import tensorflow as tf
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        X = _zscore(normal_windows)[..., None]

        self.encoder = build_encoder(self.win_len, self.embed_dim)
        proj = _projection_head(self.embed_dim)
        opt = tf.keras.optimizers.Adam(1e-3)
        vars = self.encoder.trainable_variables + proj.trainable_variables

        @tf.function
        def step(xb):
            with tf.GradientTape() as tape:
                z = tf.concat([proj(self.encoder(self._augment(xb))),
                               proj(self.encoder(self._augment(xb)))], axis=0)
                loss = self._nt_xent(z)
            opt.apply_gradients(zip(tape.gradient(loss, vars), vars))
            return loss

        n = len(X)
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            losses = []
            for s in range(0, n - 1, self.batch):
                xb = tf.gather(X, idx[s:s + self.batch])
                if xb.shape[0] < 4:
                    continue
                losses.append(float(step(xb)))
            if verbose and (ep % 10 == 0 or ep == self.epochs - 1):
                print(f"    ssl epoch {ep:3d}  loss {np.mean(losses):.3f}")

        # fit a Gaussian to the normal embeddings
        Z = self.encoder.predict(X, batch_size=256, verbose=0)
        self.mu_ = Z.mean(axis=0)
        self.sd_ = Z.std(axis=0) + 1e-8
        Zs = (Z - self.mu_) / self.sd_
        cov = np.cov(Zs, rowvar=False) + self.reg * np.eye(Zs.shape[1])
        self.inv_cov_ = np.linalg.pinv(cov)
        self.center_ = Zs.mean(axis=0)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        X = _zscore(windows)[..., None]
        Z = self.encoder.predict(X, batch_size=256, verbose=0)
        Zs = (Z - self.mu_) / self.sd_
        d = Zs - self.center_
        return np.einsum("ij,jk,ik->i", d, self.inv_cov_, d)   # squared Mahalanobis
