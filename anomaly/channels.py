"""what is each wrist channel worth? BVP vs EDA vs skin temperature vs motion.

The project reads WESAD's wrist BVP and nothing else. The same pickles carry the
E4's EDA, skin temperature and accelerometer, and the stress literature says
EDA in particular is the channel that moves the needle. Before spending a
dirham on a sensor, this measures it -- on the project's own pipeline, with the
project's own metrics, subject-wise.

Every channel becomes a small per-window feature vector; the detector is the
project's statistical one-class baseline (Mahalanobis on calm), so channels are
compared on equal terms rather than each with a different model. Three regimes
per channel set, matching `anomaly.calibrate`:

    zero-shot    fit on the OTHER subjects' calm -- a stranger's model
    calibrated   fit on the first half of THIS subject's calm -- onboarding
    supervised   logistic regression on the other subjects' calm AND stress,
                 labels and all: the ceiling, for context, not the method

all three are scored on the same test set (the second half of the subject's
calm + all their stress), so the columns are comparable with each other and
with the calibration table in RESULTS.md. window features are cached after the
first run; the pickles are ~13 GB and take a few minutes to read once.

    python -m anomaly.channels
    python -m anomaly.channels --sets BVP EDA BVP+EDA
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.signal import find_peaks

from .baseline import MahalanobisDetector
from .features import extract_batch
from .metrics import aggregate, summarize
from .wesad import (FS, FS_ACC, FS_EDA, SUBJECTS, USABLE, WESAD_DIR,
                    load_subject)

NORMAL, POSITIVE = 1, 2
CACHE_DIR = os.path.join(WESAD_DIR, "_harness_cache")
CHANNELS = ("BVP", "EDA", "TEMP", "ACC")
DEFAULT_SETS = ("BVP", "EDA", "TEMP", "ACC",
                "BVP+EDA", "BVP+TEMP", "BVP+ACC", "EDA+TEMP",
                "BVP+EDA+TEMP", "BVP+EDA+TEMP+ACC")


# ------------------------------------------------------------- windowing
def window_starts(labels: np.ndarray, win_sec: float, step_sec: float):
    """start sample (at FS) of every window that sits inside one usable
    condition -- the same purity rule make_windows() applies, kept separate so
    the SAME grid can slice channels at other sample rates."""
    win, step = int(round(win_sec * FS)), max(1, int(round(step_sec * FS)))
    out = []
    for s in range(0, len(labels) - win + 1, step):
        u = np.unique(labels[s:s + win])
        if u.size == 1 and int(u[0]) in (NORMAL, POSITIVE):
            out.append((s, int(u[0])))
    return out


def slice_at(x: np.ndarray, t_sec: float, win_sec: float, fs: int):
    a = int(round(t_sec * fs)); b = a + int(round(win_sec * fs))
    return x[a:b] if b <= len(x) else None


# -------------------------------------------------------------- features
def eda_features(w: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    """tonic level and trend, plus the phasic response count.

    skin conductance responses are the sympathetic bursts the literature keys
    on: bumps of >~0.01 uS riding on the slow tonic level. counted after taking
    a linear trend off, so a rising baseline is not mistaken for responses."""
    t = np.arange(len(w)) / fs
    slope, icpt = np.polyfit(t, w, 1)
    phasic = w - (slope * t + icpt)
    pk, _ = find_peaks(phasic, prominence=0.01, distance=int(fs * 1.0))
    d = np.diff(w)
    return np.array([w.mean(), slope, w.std(), w.max() - w.min(),
                     len(pk) / (len(w) / fs) * 60.0,           # SCRs per minute
                     d[d > 0].sum() / (len(w) / fs)],           # rise per second
                    dtype=np.float32)


def temp_features(w: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    t = np.arange(len(w)) / fs
    slope = np.polyfit(t, w, 1)[0]
    return np.array([w.mean(), slope, w.std(), w.max() - w.min()], dtype=np.float32)


def acc_features(w: np.ndarray, fs: int = FS_ACC) -> np.ndarray:
    m = np.sqrt((w.astype(np.float64) ** 2).sum(axis=1))
    return np.array([m.mean(), m.std(), np.abs(np.diff(m)).mean(), m.max(),
                     w[:, 0].std(), w[:, 1].std(), w[:, 2].std()], dtype=np.float32)


def build_features(subject: str, win_sec: float, step_sec: float) -> dict:
    """one subject -> {channel: (n_win x n_feat)}, plus cond (n_win,)."""
    d = load_subject(subject, with_acc=True, with_eda=True, with_temp=True)
    starts = window_starts(d["labels"], win_sec, step_sec)
    F = {c: [] for c in CHANNELS}
    cond = []
    bvp_wins = []
    for s, c in starts:
        t = s / FS
        e = slice_at(d["eda"], t, win_sec, FS_EDA)
        tp = slice_at(d["temp"], t, win_sec, FS_EDA)
        a = slice_at(d["acc"], t, win_sec, FS_ACC)
        b = d["bvp"][s:s + int(round(win_sec * FS))]
        if e is None or tp is None or a is None:
            continue                       # the odd channel a few samples short
        bvp_wins.append(b)
        F["EDA"].append(eda_features(e))
        F["TEMP"].append(temp_features(tp))
        F["ACC"].append(acc_features(a))
        cond.append(c)
    F["BVP"] = extract_batch(np.asarray(bvp_wins, np.float32), FS)
    out = {c: np.asarray(F[c], np.float32) for c in CHANNELS}
    out["cond"] = np.asarray(cond, np.int8)
    return out


def load_features(subjects, win_sec, step_sec) -> dict:
    path = os.path.join(CACHE_DIR, f"channels_win{win_sec:g}_step{step_sec:g}.npz")
    cache = {}
    if os.path.exists(path):
        z = np.load(path)
        cache = {k: z[k] for k in z.files}
    data, new = {}, False
    for s in subjects:
        keys = [f"{s}_{c}" for c in CHANNELS] + [f"{s}_cond"]
        if all(k in cache for k in keys):
            data[s] = {c: cache[f"{s}_{c}"] for c in CHANNELS}
            data[s]["cond"] = cache[f"{s}_cond"]
            src = "cached"
        else:
            t0 = time.time()
            data[s] = build_features(s, win_sec, step_sec)
            for c in CHANNELS:
                cache[f"{s}_{c}"] = data[s][c]
            cache[f"{s}_cond"] = data[s]["cond"]
            new, src = True, f"read {time.time() - t0:4.0f}s"
        n1 = int((data[s]["cond"] == NORMAL).sum())
        n2 = int((data[s]["cond"] == POSITIVE).sum())
        print(f"  {src:10s} {s}: {n1} calm / {n2} stress windows", flush=True)
    if new:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, **cache)
    return data


# ---------------------------------------------------------------- models
def supervised(Ftr, ytr, Fte):
    """logistic regression, standardised, median-imputed. the labelled ceiling."""
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(Ftr, axis=0)
    Ftr = np.where(np.isnan(Ftr), med, Ftr)
    Fte = np.where(np.isnan(Fte), med, Fte)
    mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit((Ftr - mu) / sd, ytr)
    return clf.decision_function((Fte - mu) / sd)


def run(data, sets, calib_frac: float):
    subjects = list(data)
    results = {}
    for name in sets:
        chans = name.split("+")
        rows = {"zero-shot": [], "calibrated": [], "supervised": []}
        for test in subjects:
            cat = lambda s: np.hstack([data[s][c] for c in chans])
            Fte_all, cte = cat(test), data[test]["cond"]
            Fcalm, Fstr = Fte_all[cte == NORMAL], Fte_all[cte == POSITIVE]
            k = max(5, int(len(Fcalm) * calib_frac))
            Fcal, Ftest_calm = Fcalm[:k], Fcalm[k:]
            Fte = np.vstack([Ftest_calm, Fstr])
            y = np.r_[np.zeros(len(Ftest_calm)), np.ones(len(Fstr))].astype(int)

            others = [s for s in subjects if s != test]
            Fo_calm = np.vstack([cat(s)[data[s]["cond"] == NORMAL] for s in others])
            Fo_all = np.vstack([cat(s) for s in others])
            yo = np.concatenate([(data[s]["cond"] == POSITIVE).astype(int) for s in others])

            sc = {
                "zero-shot":  MahalanobisDetector().fit_features(Fo_calm).score_features(Fte),
                "calibrated": MahalanobisDetector().fit_features(Fcal).score_features(Fte),
                "supervised": supervised(Fo_all, yo, Fte),
            }
            for r, v in sc.items():
                m = summarize(y, v); m["subject"] = test
                rows[r].append(m)
        results[name] = {r: aggregate(rows[r]) for r in rows}
        results[name]["_per_subject"] = rows
        z, c, s_ = (results[name][r]["pr_auc"][0] for r in ("zero-shot", "calibrated", "supervised"))
        print(f"  {name:18s} PR-AUC  zero-shot {z:.3f}  calibrated {c:.3f}  supervised {s_:.3f}",
              flush=True)
    return results


def table(results, sets) -> str:
    hdr = ("| channels | zero-shot PR-AUC | calibrated PR-AUC | supervised PR-AUC | "
           "zero-shot ROC | calibrated ROC | supervised ROC | calibrated recall@90 |\n"
           "|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for name in sets:
        R = results[name]
        f = lambda r, k: "%.3f ± %.3f" % R[r][k]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, f("zero-shot", "pr_auc"), f("calibrated", "pr_auc"),
            f("supervised", "pr_auc"), f("zero-shot", "roc_auc"),
            f("calibrated", "roc_auc"), f("supervised", "roc_auc"),
            f("calibrated", "recall@90spec")))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--out", default=None, help="write the markdown table here")
    a = ap.parse_args(argv)

    print(f"features: win={a.win:g}s step={a.step:g}s  subjects={len(SUBJECTS)}\nloading…")
    data = load_features(SUBJECTS, a.win, a.step)
    print("\nLOSO, per channel set:")
    results = run(data, a.sets, a.calib_frac)
    md = table(results, a.sets)
    print("\n" + md)

    # the per-subject spread for the two sets that matter most
    for name in ("BVP", "EDA", "BVP+EDA"):
        if name in results:
            v = sorted((r["pr_auc"], r["subject"])
                       for r in results[name]["_per_subject"]["calibrated"])
            print(f"\n{name} calibrated, per subject (PR-AUC): "
                  + "  ".join(f"{s}={p:.2f}" for p, s in v))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
