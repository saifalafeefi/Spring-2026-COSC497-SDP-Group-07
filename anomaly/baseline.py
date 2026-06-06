"""statistical one-class baseline — the number the autoencoder must beat.

fit a Gaussian to the NORMAL feature distribution, score new windows by their
Mahalanobis distance from it. simple, transparent, no training loop — exactly
what a baseline should be.
"""
from __future__ import annotations

import numpy as np

from .features import extract_batch


class MahalanobisDetector:
    """fit on normal feature vectors; score = distance from the normal Gaussian.

    NaN features (e.g. windows with too few detectable pulses) are imputed with
    the train-set median before fitting, so they never poison the covariance.
    """

    def __init__(self, fs: int = 64, reg: float = 1e-3):
        self.fs = fs
        self.reg = reg

    def _impute_standardize(self, F, fit):
        if fit:
            self.median_ = np.nanmedian(F, axis=0)
        F = np.where(np.isnan(F), self.median_, F)
        if fit:
            self.mu_ = F.mean(axis=0)
            self.sd_ = F.std(axis=0) + 1e-8
        return (F - self.mu_) / self.sd_

    def fit(self, normal_windows: np.ndarray):
        F = self._impute_standardize(extract_batch(normal_windows, self.fs), fit=True)
        cov = np.cov(F, rowvar=False) + self.reg * np.eye(F.shape[1])
        self.inv_cov_ = np.linalg.pinv(cov)
        self.center_ = F.mean(axis=0)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """anomaly score per window (higher = more deviant from normal)."""
        F = self._impute_standardize(extract_batch(windows, self.fs), fit=False)
        d = F - self.center_
        return np.einsum("ij,jk,ik->i", d, self.inv_cov_, d)   # squared Mahalanobis
