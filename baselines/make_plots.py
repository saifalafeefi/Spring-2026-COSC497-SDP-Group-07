"""Generate result PNGs in the repo root.

Outputs (placed in repo root):
  01_data_overview.png       Dataset class balance + example PPG signals
  02_training_progress.png   Final model training curves
  03_model_performance.png   Final model confusion matrix + per-class metrics

Run from repo root:  python3 baselines/make_plots.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CLASSES, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(HERE, "runs")

FINAL_PRESET = "phase_a"

CLASS_DISPLAY = {"cardiac": "Cardiac", "non_cardiac": "Non-Cardiac", "occlusion": "Occlusion"}
CLASS_COLORS = {"cardiac": "#2E86AB", "non_cardiac": "#A23B72", "occlusion": "#E07A5F"}
PRIMARY = "#2E86AB"
ACCENT = "#E07A5F"


def _save(fig, name):
    path = os.path.join(ROOT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _final_run() -> str:
    candidates = sorted(d for d in os.listdir(RUNS_DIR) if d.endswith(f"_{FINAL_PRESET}"))
    if not candidates:
        raise FileNotFoundError(
            f"No final-model run found in {RUNS_DIR}. "
            f"Run `python3 baselines/train.py --preset {FINAL_PRESET}` first.")
    return os.path.join(RUNS_DIR, candidates[-1])


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------

def plot_01_data_overview():
    print("Building 01_data_overview.png ...")
    ds = load_dataset(placement="all", zscore=False)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.42, wspace=0.3)

    # Class distribution
    ax = fig.add_subplot(gs[0, 0])
    counts = np.bincount(ds.y, minlength=3)
    labels = [CLASS_DISPLAY[c] for c in CLASSES]
    bars = ax.bar(labels, counts, color=[CLASS_COLORS[c] for c in CLASSES])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 200, f"{c:,}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of windows")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.tick_params(axis="x", labelsize=10)

    # Per-participant volume
    ax = fig.add_subplot(gs[0, 1])
    pcs = {}
    for g in ds.groups:
        pcs[g] = pcs.get(g, 0) + 1
    parts = sorted(pcs.keys(), key=lambda x: -pcs[x])
    ax.bar(range(len(parts)), [pcs[p] for p in parts], color=PRIMARY)
    ax.set_xticks([])
    ax.set_title("Windows per Participant", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of windows")
    ax.set_xlabel("Participants (n=32)")

    # Placement distribution
    ax = fig.add_subplot(gs[0, 2])
    placement_pretty = {"MFT_LP": "Fingertip (M)", "IFT_LP": "Fingertip (I)",
                        "IFB_LP": "Finger Base", "WI_LP": "Wrist (Inner)",
                        "WO_LP": "Wrist (Outer)", "OB": "Off-Body"}
    placements, pc = np.unique(ds.placements, return_counts=True)
    order = np.argsort(-pc)
    ax.bar([placement_pretty.get(p, p) for p in placements[order]], pc[order],
           color=PRIMARY)
    ax.set_title("Sensor Placement Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of windows")
    ax.tick_params(axis="x", labelsize=9, rotation=30)

    # Example signals per class
    for i, cls in enumerate(CLASSES):
        ax = fig.add_subplot(gs[1, i])
        cls_idx = np.where(ds.y == i)[0]
        picks = rng.choice(cls_idx, size=min(3, len(cls_idx)), replace=False)
        for j, k in enumerate(picks):
            sig = ds.X[k]
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            t = np.arange(len(sig)) / 50.0
            ax.plot(t, sig + j * 5, color=CLASS_COLORS[cls], linewidth=0.9)
        ax.set_title(f"Representative {CLASS_DISPLAY[cls]} Signals",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_xlim(0, 10.24)

    fig.suptitle("PPG Dataset Overview",
                 fontsize=15, fontweight="bold", y=0.99)
    _save(fig, "01_data_overview.png")


def plot_02_training_progress():
    print("Building 02_training_progress.png ...")
    run = _final_run()
    history_path = os.path.join(run, "history.json")
    if not os.path.exists(history_path):
        print(f"  no history.json in {run}; skipping")
        return
    h = _load_json(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(h.get("accuracy", []), label="Training", color=PRIMARY,
            linestyle="--", linewidth=1.5)
    ax.plot(h.get("val_accuracy", []), label="Validation", color=ACCENT, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(h.get("loss", []), label="Training", color=PRIMARY,
            linestyle="--", linewidth=1.5)
    ax.plot(h.get("val_loss", []), label="Validation", color=ACCENT, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Model Training Progress", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_training_progress.png")


def plot_03_model_performance():
    print("Building 03_model_performance.png ...")
    run = _final_run()
    results_path = os.path.join(run, "results.json")
    if not os.path.exists(results_path):
        print(f"  no results.json in {run}; skipping")
        return
    r = _load_json(results_path)

    cm = np.array(r["confusion_matrix"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    # Confusion matrix
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    labels = [CLASS_DISPLAY[c] for c in CLASSES]
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            color_t = "white" if cm_pct[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color_t)

    # Per-class metrics
    ax = fig.add_subplot(gs[0, 1])
    metrics = ["Precision", "Recall", "F1-Score"]
    keys = ["per_class_precision", "per_class_recall", "per_class_f1"]
    n_metrics = len(metrics)
    n_classes = len(CLASSES)
    x = np.arange(n_metrics)
    bar_w = 0.25
    for ci, cls in enumerate(CLASSES):
        vals = [r[k][cls] for k in keys]
        offsets = x + (ci - 1) * bar_w
        bars = ax.bar(offsets, vals, bar_w,
                      label=CLASS_DISPLAY[cls], color=CLASS_COLORS[cls])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    overall = (f"Accuracy: {r['test_accuracy']*100:.1f}%   |   "
               f"Macro F1: {r['macro_f1']:.3f}")
    fig.suptitle(f"Model Performance\n{overall}",
                 fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "03_model_performance.png")


def main():
    plot_01_data_overview()
    plot_02_training_progress()
    plot_03_model_performance()
    print("\nDone.")


if __name__ == "__main__":
    main()
