"""generate result figures for the one-class anomaly detector (WESAD).

reads the windowed cache (built by run.py) + the saved autoencoder + the measured
leave-one-subject-out numbers, and writes three PNGs into anomaly/:

  fig1_wesad_overview.png   class balance, example calm vs stress BVP, per-subject counts
  fig2_recon_separation.png reconstruction-error histograms (calm vs stress) + threshold
  fig3_results.png          model comparison (mean +/- std) + per-subject PR-AUC spread

    python3 -m anomaly.make_plots
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .wesad import WESAD_DIR, SUBJECTS, FS

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WESAD_DIR, "_harness_cache", "win60_step5.npz")

# measured leave-one-subject-out PR-AUC per subject (from the full run / RESULTS.md)
PR = {
    "baseline": [0.439, 0.234, 0.290, 0.837, 0.762, 0.773, 0.412, 0.478, 0.370,
                 0.929, 0.723, 0.936, 0.478, 0.980, 0.880],
    "ae":       [0.385, 0.537, 0.989, 0.923, 0.304, 0.596, 0.479, 0.843, 0.702,
                 0.730, 0.615, 0.970, 0.313, 0.655, 0.955],
    "ssl":      [0.709, 0.367, 0.573, 0.809, 0.294, 0.602, 0.778, 0.567, 0.423,
                 0.903, 0.622, 0.999, 0.697, 0.962, 0.826],
}
# (pr_mean, pr_std, recall_mean, recall_std)
SUMMARY = {"baseline": (0.635, 0.249, 0.437, 0.326),
           "ae":       (0.667, 0.228, 0.507, 0.328),
           "ssl":      (0.676, 0.204, 0.502, 0.286)}
NAMES = {"baseline": "baseline", "ae": "autoencoder (O1)", "ssl": "SSL (O2)"}
COLORS = {"baseline": "#8b98a5", "ae": "#2E86AB", "ssl": "#E07A5F"}


def load_cache():
    z = np.load(CACHE)
    calm, stress, counts = [], [], {}
    for s in SUBJECTS:
        X, c = z[f"{s}_X"], z[f"{s}_c"]
        counts[s] = (int((c == 1).sum()), int((c == 2).sum()))
        calm.append(X[c == 1]); stress.append(X[c == 2])
    return np.concatenate(calm), np.concatenate(stress), counts


def fig1_overview(calm, stress, counts):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(["calm", "stress"], [len(calm), len(stress)],
              color=["#27ae60", "#c0392b"])
    ax[0].set_title("WESAD wrist-BVP windows (60 s)"); ax[0].set_ylabel("windows")

    t = np.arange(calm.shape[1]) / FS
    ax[1].plot(t, calm[len(calm)//2], color="#27ae60", lw=.8, label="calm")
    ax[1].plot(t, stress[len(stress)//2] + 600, color="#c0392b", lw=.8, label="stress (+offset)")
    ax[1].set_title("example BVP windows"); ax[1].set_xlabel("seconds"); ax[1].legend()

    subs = list(counts); nc = [counts[s][0] for s in subs]; ns = [counts[s][1] for s in subs]
    x = np.arange(len(subs))
    ax[2].bar(x, nc, color="#27ae60", label="calm")
    ax[2].bar(x, ns, bottom=nc, color="#c0392b", label="stress")
    ax[2].set_xticks(x); ax[2].set_xticklabels(subs, rotation=90, fontsize=7)
    ax[2].set_title("windows per subject"); ax[2].legend()
    fig.tight_layout(); _save(fig, "fig1_wesad_overview.png")


def fig2_recon(calm, stress):
    from .infer import LiveAnomalyDetector
    det = LiveAnomalyDetector()

    def score(X):
        Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        r = det.model.predict(Xz[..., None], batch_size=256, verbose=0)
        return np.mean((r - Xz[..., None]) ** 2, axis=(1, 2))

    sc, ss = score(calm), score(stress)
    hi = np.quantile(np.concatenate([sc, ss]), 0.99)
    bins = np.linspace(0, hi, 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sc, bins=bins, color="#27ae60", alpha=.6, density=True, label="calm")
    ax.hist(ss, bins=bins, color="#c0392b", alpha=.6, density=True, label="stress")
    ax.axvline(det.threshold, color="k", ls="--", lw=1.5,
               label=f"flag threshold ({det.threshold:.4f})")
    ax.set_title("autoencoder reconstruction error: calm vs stress")
    ax.set_xlabel("reconstruction error (MSE)"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); _save(fig, "fig2_recon_separation.png")


def fig3_results():
    models = ["baseline", "ae", "ssl"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(models)); w = 0.35
    pr_m = [SUMMARY[m][0] for m in models]; pr_s = [SUMMARY[m][1] for m in models]
    rc_m = [SUMMARY[m][2] for m in models]; rc_s = [SUMMARY[m][3] for m in models]
    ax[0].bar(x - w/2, pr_m, w, yerr=pr_s, capsize=4, color="#2E86AB", label="PR-AUC")
    ax[0].bar(x + w/2, rc_m, w, yerr=rc_s, capsize=4, color="#E07A5F", label="recall@90% spec")
    ax[0].set_xticks(x); ax[0].set_xticklabels([NAMES[m] for m in models])
    ax[0].set_ylim(0, 1); ax[0].set_title("leave-one-subject-out (mean +/- std)")
    ax[0].legend()

    for i, m in enumerate(models):
        jitter = (np.random.RandomState(0).rand(len(PR[m])) - .5) * .25
        ax[1].scatter(np.full(len(PR[m]), i) + jitter, PR[m],
                      color=COLORS[m], alpha=.8, s=30)
        ax[1].plot([i-.2, i+.2], [np.mean(PR[m])]*2, color="k", lw=2)
    ax[1].set_xticks(range(len(models))); ax[1].set_xticklabels([NAMES[m] for m in models])
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("PR-AUC (one dot = one subject)")
    ax[1].set_title("per-subject spread = the domain-transfer story")
    fig.tight_layout(); _save(fig, "fig3_results.png")


def _save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"cache not found: {CACHE}\nrun `python3 -m anomaly.run --model baseline` first.")
    calm, stress, counts = load_cache()
    print(f"loaded {len(calm)} calm + {len(stress)} stress windows")
    fig1_overview(calm, stress, counts)
    fig2_recon(calm, stress)
    fig3_results()
    print("done.")


if __name__ == "__main__":
    main()
