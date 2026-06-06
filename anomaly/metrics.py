"""sensitivity-first metrics — locked before any modelling.

higher score = more anomalous = more likely the positive class (stress).
pure numpy so there's no hidden sklearn-version drift in the headline numbers.

the two we pre-commit to:
  • PR-AUC (average precision) — robust to the heavy class imbalance (stress is rare)
  • recall @ 90% specificity — "of real stress, how much do we catch while keeping
    false alarms on normal at ≤10%" — the number a screening tool lives or dies by
"""
from __future__ import annotations

import numpy as np


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """average precision (area under the precision-recall curve)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    P = int(y.sum())
    if P == 0 or P == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")     # high score first
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / P
    # sum precision over the recall steps where a true positive is added
    return float(np.sum(precision[y == 1]) / P)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """area under the ROC curve, via the rank (Mann-Whitney U) identity."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    P, N = int(y.sum()), int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    ranks = s.argsort().argsort() + 1            # average ties roughly; fine here
    return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def recall_at_specificity(y_true: np.ndarray, scores: np.ndarray,
                          specificity: float = 0.90) -> float:
    """recall (sensitivity) at the threshold giving the target specificity.

    threshold = the `specificity` quantile of the NORMAL scores (so that fraction
    of normals fall below it = specificity); recall = fraction of positives above.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, specificity)
    return float(np.mean(pos >= thr))


def summarize(y_true: np.ndarray, scores: np.ndarray, spec: float = 0.90) -> dict:
    return {
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        f"recall@{int(spec*100)}spec": recall_at_specificity(y_true, scores, spec),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }


def aggregate(per_subject: list[dict]) -> dict:
    """mean ± std across per-subject metric dicts (nan-safe)."""
    out = {}
    keys = [k for k, v in per_subject[0].items() if isinstance(v, float)]
    for k in keys:
        vals = np.array([d[k] for d in per_subject], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out
