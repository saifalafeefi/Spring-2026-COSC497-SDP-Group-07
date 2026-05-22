"""Centralized training entry point.

Pick a preset, optionally override knobs, run. Every run drops a timestamped
folder under baselines/runs/ with config, model, results, history, and plots.

Examples:
  python3 baselines/train.py --preset phase_a
  python3 baselines/train.py --preset phase_a
  python3 baselines/train.py --preset phase_a --epochs 200 --dropout 0.3
  python3 baselines/train.py --list-presets
  python3 baselines/train.py --preset phase_a --print-config

For full description of every config knob, see baselines/configs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, fields

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight

import augment as aug_mod
import models as model_mod
from configs import PRESETS, Config, get_preset
from data import (CLASSES, load_dataset, participant_split, stratified_split)
from features import extract_features, standardize
from losses import make_loss


# --------------------------- argument handling ------------------------------

def add_config_overrides(parser: argparse.ArgumentParser):
    """Auto-generate one CLI flag per Config field, so any knob can be overridden."""
    for f in fields(Config):
        flag = f"--{f.name.replace('_', '-')}"
        if f.type is bool or f.default is True or f.default is False:
            parser.add_argument(flag, dest=f.name, type=lambda s: s.lower() in {"1", "true", "yes", "y"})
        else:
            parser.add_argument(flag, dest=f.name, type=type(f.default), default=None)


def apply_overrides(cfg: Config, args) -> Config:
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


# ------------------------------ data plumbing -------------------------------

def get_split(ds, cfg: Config):
    if cfg.split == "stratified":
        return stratified_split(ds, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)
    if cfg.split == "participant_disjoint":
        return participant_split(ds, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)
    raise ValueError(f"unknown split: {cfg.split}")


def get_class_weight(y_train, cfg: Config):
    if cfg.class_weight == "none":
        return None
    counts = np.bincount(y_train, minlength=cfg.n_classes)
    total = counts.sum()
    return {i: float(total / (len(counts) * c)) if c > 0 else 0.0
            for i, c in enumerate(counts)}


def make_lr_schedule(cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.schedule == "constant":
        return cfg.learning_rate
    if cfg.schedule == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=total_steps,
            alpha=0.02)
    if cfg.schedule == "cosine_warmup":
        warmup_steps = int(total_steps * cfg.warmup_frac)
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=0.02,
            warmup_target=cfg.learning_rate,
            warmup_steps=warmup_steps)
    if cfg.schedule == "reduce_on_plateau":
        return cfg.learning_rate  # actual reduction handled in callbacks
    raise ValueError(f"unknown schedule: {cfg.schedule}")


def make_tf_dataset(X, y, F, training: bool, cfg: Config):
    X = X.astype(np.float32)[..., None]
    y = y.astype(np.int64)
    inputs = (X,) if F is None else (X, F.astype(np.float32))

    ds = tf.data.Dataset.from_tensor_slices((inputs, y) if F is not None else (X, y))
    if training:
        ds = ds.shuffle(4096, seed=cfg.seed)
        if any([cfg.augment_shift, cfg.augment_scale, cfg.augment_noise, cfg.augment_time_warp]):
            augmenter = aug_mod.make_augmenter(
                do_shift=cfg.augment_shift, shift_max=cfg.shift_max,
                do_scale=cfg.augment_scale, scale_range=(cfg.scale_min, cfg.scale_max),
                do_noise=cfg.augment_noise, noise_std=cfg.noise_std,
                do_time_warp=cfg.augment_time_warp, warp_strength=cfg.warp_strength)
            if F is None:
                ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                def aug_wrap(inputs, label):
                    sig, feat = inputs
                    sig, label = augmenter(sig, label)
                    return (sig, feat), label
                ds = ds.map(aug_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


# -------------------------------- plotting ----------------------------------

def save_plots(history, cm, cfg, run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["accuracy"], label="train", linestyle="--")
    axes[0].plot(history["val_accuracy"], label="val", linewidth=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"Accuracy — {cfg.name}"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["loss"], label="train", linestyle="--")
    axes[1].plot(history["val_loss"], label="val", linewidth=2)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].set_title(f"Loss — {cfg.name}"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=20); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix — {cfg.name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = "white" if cm_pct[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)", ha="center", va="center", color=c)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- main -------------------------------------

def run(cfg: Config) -> dict:
    # set seeds
    tf.keras.utils.set_random_seed(cfg.seed)

    # output folder
    here = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(here, cfg.out_dir, f"{timestamp}_{cfg.name}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fp:
        json.dump(asdict(cfg), fp, indent=2)
    print(f"\nRun dir: {run_dir}")
    print(f"Config:  {cfg.name} | model={cfg.model} | epochs={cfg.epochs} | "
          f"loss={cfg.loss} | split={cfg.split} | placement={cfg.placement}")

    # data
    ds = load_dataset(placement=cfg.placement, zscore=True)
    tr, va, te = get_split(ds, cfg)
    print(f"Train={len(tr)}  Val={len(va)}  Test={len(te)}  "
          f"Classes(train)={np.bincount(ds.y[tr]).tolist()}")

    # features (only if model needs them)
    model = model_mod.build(cfg)
    use_feats = model_mod.needs_features(model)
    if use_feats:
        # NOTE: features computed on raw (non-z-scored) data
        ds_raw = load_dataset(placement=cfg.placement, zscore=False)
        F_all = extract_features(ds_raw.X)
        F_tr, F_va, F_te = standardize(F_all[tr], F_all[va], F_all[te])
        cfg.n_features = F_tr.shape[1]
    else:
        F_tr = F_va = F_te = None

    tr_ds = make_tf_dataset(ds.X[tr], ds.y[tr], F_tr, training=True,  cfg=cfg)
    va_ds = make_tf_dataset(ds.X[va], ds.y[va], F_va, training=False, cfg=cfg)
    if use_feats:
        te_inputs = (ds.X[te, :, None].astype(np.float32), F_te.astype(np.float32))
    else:
        te_inputs = ds.X[te, :, None].astype(np.float32)
    te_y = ds.y[te]

    # optimizer + loss
    steps_per_epoch = (len(tr) + cfg.batch_size - 1) // cfg.batch_size
    lr = make_lr_schedule(cfg, steps_per_epoch)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = make_loss(cfg.loss, gamma=cfg.focal_gamma)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.summary(line_length=90)
    print(f"Parameters: {model.count_params():,}")

    # callbacks
    cbs = []
    if cfg.schedule == "reduce_on_plateau":
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1))
    cbs.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.early_stopping_patience,
        restore_best_weights=True, verbose=1))

    # class weights
    cw = get_class_weight(ds.y[tr], cfg)
    if cw: print(f"Class weights: {cw}")

    # train
    t0 = time.time()
    history = model.fit(tr_ds, validation_data=va_ds,
                        epochs=cfg.epochs, class_weight=cw,
                        callbacks=cbs, verbose=cfg.verbose)
    train_time = time.time() - t0
    print(f"\nTraining took {train_time:.1f}s over {len(history.history['loss'])} epochs.")

    # evaluate
    test_loss, test_acc = model.evaluate(te_inputs, te_y, verbose=0)
    y_pred = model.predict(te_inputs, verbose=0).argmax(axis=1)
    cm = confusion_matrix(te_y, y_pred, labels=list(range(cfg.n_classes)))
    report = classification_report(te_y, y_pred, target_names=CLASSES,
                                   digits=4, output_dict=True)

    print(f"\n== Test ({cfg.split}) ==")
    print(f"  accuracy={test_acc:.4f}  loss={test_loss:.4f}  macro_f1={report['macro avg']['f1-score']:.4f}")
    print(classification_report(te_y, y_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix:")
    for row in cm:
        print(f"  {row.tolist()}")

    # save artifacts
    results = {
        "config_name": cfg.name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {c: report[c]["f1-score"] for c in CLASSES},
        "per_class_precision": {c: report[c]["precision"] for c in CLASSES},
        "per_class_recall": {c: report[c]["recall"] for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "train_time_s": float(train_time),
        "epochs_run": len(history.history["loss"]),
        "n_params": int(model.count_params()),
    }
    with open(os.path.join(run_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as fp:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, fp)
    if cfg.save_model:
        model.save(os.path.join(run_dir, "model.keras"))
    if cfg.save_plots:
        save_plots(history.history, cm, cfg, run_dir)

    print(f"\nAll artifacts in: {run_dir}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="phase_a",
                   help=f"named preset (see configs.py). Available: {list(PRESETS)}")
    p.add_argument("--list-presets", action="store_true", help="list presets and exit")
    p.add_argument("--print-config", action="store_true", help="print final config and exit")
    add_config_overrides(p)
    args = p.parse_args()

    if args.list_presets:
        for name, c in PRESETS.items():
            print(f"  {name:10s}  model={c.model:14s} epochs={c.epochs:4d} loss={c.loss}")
        return

    cfg = get_preset(args.preset)
    cfg = apply_overrides(cfg, args)
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2))
        return

    run(cfg)


if __name__ == "__main__":
    main()
