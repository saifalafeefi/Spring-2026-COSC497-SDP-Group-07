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


def _runs(mask: np.ndarray):
    """maximal [start, end) index ranges where a boolean array is True."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def episode_metrics(scores: np.ndarray, y_true: np.ndarray, step_sec: float = 5.0,
                    spec: float = 0.90, k: int = 3) -> dict:
    """episode-level detection on TIME-ORDERED windows (the product's-eye view).

    a stress *episode* = a contiguous run of positive windows. it's *detected* only
    if it contains a run of >= k consecutive flagged windows — "sustained", to mirror
    the deployed debounce and kill the trivial 'any single window fires' inflation.
    the flag threshold is the same 90%-specificity point as the window metric, so the
    two are directly comparable.

    recall and false-alarms are reported TOGETHER (each is meaningless alone — loosen
    the rule and both rise):
      • ep_recall  — fraction of stress episodes detected
      • fa_per_hr  — sustained false-alarm events per hour of non-stress monitoring
      • latency_s  — median seconds from episode onset to first sustained detection

    NOTE: WESAD has ~one stress block per subject, so per-subject ep_recall is ~0/1 and
    the across-subject mean reads as "fraction of subjects whose stress we caught".
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    neg, pos = s[y == 0], s[y == 1]
    if len(pos) == 0 or len(neg) == 0:
        return {"ep_recall": float("nan"), "fa_per_hr": float("nan"), "latency_s": float("nan")}

    thr = np.quantile(neg, spec)
    events = [(a, b) for (a, b) in _runs(s >= thr) if b - a >= k]   # sustained flags
    episodes = _runs(y == 1)                                        # stress blocks

    detected, latencies = 0, []
    for ea, eb in episodes:
        first = None
        for a, b in events:
            if a < eb and b > ea:                       # event overlaps this episode
                f = max(a, ea)
                first = f if first is None else min(first, f)
        if first is not None:
            detected += 1
            latencies.append((first - ea) * step_sec)

    fa = sum(1 for a, b in events if not np.any(y[a:b] == 1))       # events touching no stress
    calm_hours = (len(neg) * step_sec) / 3600.0
    return {
        "ep_recall": float(detected / len(episodes)) if episodes else float("nan"),
        "fa_per_hr": float(fa / calm_hours) if calm_hours > 0 else float("nan"),
        "latency_s": float(np.median(latencies)) if latencies else float("nan"),
    }


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
