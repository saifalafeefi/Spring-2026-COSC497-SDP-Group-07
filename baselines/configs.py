"""Configuration system for training experiments.

EVERY KNOB you can turn is here, with a comment explaining:
  - what it does
  - what changes when you increase or decrease it
  - sensible ranges / typical values

USAGE:
  1. Pick a preset:        python3 baselines/train.py --preset phase_a
  2. Override one knob:    python3 baselines/train.py --preset phase_a --epochs 200
  3. Build a new preset:   add a new entry to PRESETS at the bottom of this file
  4. List all presets:     python3 baselines/train.py --list-presets
  5. Show full config:     python3 baselines/train.py --preset phase_a --print-config
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ============================ EXPERIMENT META ============================

    name: str = "unnamed"
    # Human-readable tag for the run folder. No spaces, no special chars.
    # Effect: only changes where results are saved. No model impact.

    seed: int = 42
    # Random seed. Same seed + same config = same numerical result.
    # Effect: change this to get a different random initialization. Run 3-5
    #   different seeds to estimate how much variance there is in your numbers.

    # ============================== DATA =====================================

    placement: str = "all"
    # Which sensor positions to keep. One of: all | fingertip | finger_base | wrist
    # Effect:
    #   - "all": uses every recording (most data, most variety, hardest)
    #   - "fingertip": only MFT+IFT placements → fewer windows but cleaner signal
    #   - "wrist": WI+WO → harder problem, but matches a wristband product
    #   - "finger_base": IFB only
    # Pick "all" for a general model, pick the placement you'll deploy on for
    # a specialist model (~3-5 pp easier accuracy).

    split: str = "participant_disjoint"
    # How to divide train/val/test. One of: stratified | participant_disjoint
    # Effect:
    #   - "stratified": random split keeping class proportions. Inflates accuracy
    #     because the same patient appears in train and test.
    #   - "participant_disjoint": no patient overlaps. This is what we report
    #     as the honest deployment number. ALWAYS use this for final results.

    test_size: float = 0.2
    # Fraction of data held out for final test. Effect: more = more reliable
    # final number but less training data. Sane range 0.15–0.25.

    val_size: float = 0.1
    # Fraction held out for validation (used for early stopping & LR schedule).
    # Sane range 0.05–0.15.

    # ============================== MODEL ====================================

    model: str = "improved_cnn"
    # Architecture. One of: tiny_cnn | improved_cnn | resnet1d | hybrid
    # Effect:
    #   - tiny_cnn:     ~9 K params. Fits anywhere. Baseline numbers.
    #   - improved_cnn: ~24 K params. Our current "best small" model.
    #   - resnet1d:     50 K–500 K params (scaled by width/depth). Maximum
    #                   accuracy when you don't care about size.
    #   - hybrid:       CNN + 18 hand-features fused. Often the strongest
    #                   accuracy-per-parameter on this data.

    width_multiplier: float = 1.0
    # Multiplies the number of filters in each conv layer.
    # Used by: resnet1d, hybrid
    # Effect: 1.0 = nominal size. 2.0 = roughly 4× parameters and FLOPs.
    # Sane range 0.5–3.0.

    depth: int = 2
    # Number of residual blocks per stage (resnet1d / hybrid).
    # Effect: more depth = more capacity but slower to train and more overfit
    # risk. Sane range 1–6.

    dropout: float = 0.2
    # Fraction of activations randomly zeroed during training (regularization).
    # Effect:
    #   - 0.0 → no dropout, model can memorize easily
    #   - 0.1–0.2 → light regularization (usually best)
    #   - 0.3–0.5 → strong regularization, may underfit
    #   - >0.5 → almost always too much.

    n_classes: int = 3
    # Number of output classes. Don't change unless you reframe the problem
    # (e.g., binary "alarm vs no alarm").

    input_len: int = 512
    # Samples per window. The dataset is fixed at 512 (10.24 s @ 50 Hz).
    # Don't change unless you re-segment the raw data.

    n_features: int = 18
    # Length of the engineered feature vector (used by hybrid model).
    # See features.py — change only if you add/remove features there.

    # ============================ LOSS / WEIGHTING ===========================

    loss: str = "cross_entropy"
    # Training loss. One of: cross_entropy | focal
    # Effect:
    #   - cross_entropy: standard. Pair with class_weight=inverse for imbalance.
    #   - focal: emphasizes hard examples. Often improves the rare class.

    focal_gamma: float = 2.0
    # Focusing parameter for focal loss.
    # Effect: 0 → plain CE, 1 → mild, 2 → typical, 3+ → very aggressive.

    class_weight: str = "inverse"
    # How to weight class loss contributions. One of: none | inverse
    # Effect:
    #   - none: trains to maximize raw accuracy → ignores rare class
    #   - inverse: weights inversely proportional to class frequency →
    #     forces the model to care about occlusion. REQUIRED for this dataset.

    # ============================== TRAINING =================================

    epochs: int = 80
    # Max training epochs (passes over the data).
    # Effect: more epochs = more learning, but diminishing returns + overfit
    # risk after a point. Early stopping caps actual run length.
    # Sane range 40–200.

    batch_size: int = 64
    # Number of windows per gradient step.
    # Effect:
    #   - Small (16–32): noisier gradients, sometimes better generalization
    #   - Medium (64–128): typical
    #   - Large (256+): smoother gradients, faster epochs, may need higher LR
    # On CPU, keep ≤128 or epochs get slow.

    learning_rate: float = 1e-3
    # Initial learning rate for Adam.
    # Effect: too high → training diverges (loss explodes); too low → very slow.
    # Sane range 1e-4 to 5e-3. Combine with a schedule (below).

    schedule: str = "cosine"
    # Learning-rate schedule. One of:
    #   - constant: lr stays at initial value
    #   - cosine: smoothly decays initial_lr → initial_lr*0.02 over training
    #   - cosine_warmup: linearly warms up first 5% of steps, then cosine
    #   - reduce_on_plateau: halves lr when val_loss stops improving
    # Effect: cosine and cosine_warmup tend to give the best final numbers on
    #   small-to-medium datasets. reduce_on_plateau is safer if you're unsure.

    warmup_frac: float = 0.05
    # Fraction of total steps spent in warmup (used by cosine_warmup).
    # Effect: small datasets benefit from short warmup (0.02–0.1).

    early_stopping_patience: int = 15
    # Stop training if val_loss doesn't improve for this many epochs.
    # Effect: lower = more aggressive stopping (may end too early on noisy runs).
    # Higher = trains longer at the risk of slight overfit.

    # ============================ AUGMENTATION ===============================

    augment_shift: bool = True
    # Random time-shift augmentation (cyclic roll within the window).
    # Effect: forces the model to be invariant to where the heartbeat lands
    #   in the window. Almost always helps. Disable only to check its impact.

    shift_max: int = 32
    # Maximum shift in samples (32 samples = 0.64 s at 50 Hz).
    # Effect: bigger = more invariance but can shift important features off.

    augment_scale: bool = True
    # Random amplitude-scale augmentation.
    # Effect: mimics sensor sensitivity drift between users. Usually helps.

    scale_min: float = 0.85
    scale_max: float = 1.15
    # Multiplier range. (0.85, 1.15) means ±15% amplitude. Sane: ±10–25%.

    augment_noise: bool = True
    # Add Gaussian noise to the signal.
    # Effect: mimics electrical/sensor noise; improves robustness.

    noise_std: float = 0.05
    # Standard deviation of the noise (relative to z-scored signal).
    # Effect: 0.02 → light, 0.05 → moderate, 0.1+ → heavy. >0.2 hurts.

    augment_time_warp: bool = False
    # Random time-warping (stretches/squeezes the window in time).
    # Effect: mimics natural heart-rate variability. Off by default because
    #   it's expensive to compute and only sometimes helps.

    warp_strength: float = 0.10
    # Max relative speed change. 0.10 = ±10% pulse-rate variation.

    augment_mixup: bool = False
    # Mixup augmentation: blend two windows + their labels.
    # Effect: powerful regularizer, but slows convergence. Try gamma 0.1–0.4.

    mixup_alpha: float = 0.2
    # Beta distribution parameter for mixup. 0 = off. Sane range 0.1–0.4.

    # =============================== I/O =====================================

    out_dir: str = "runs"
    # Where run folders are created (under baselines/).
    # Effect: change to organize experiment families.

    save_model: bool = True
    # Save the trained Keras model file. Set False to save disk space.

    save_plots: bool = True
    # Save training-curve and confusion-matrix PNGs in the run folder.

    verbose: int = 2
    # 0 = silent, 1 = progress bar, 2 = one line per epoch.


# =============================== PRESETS =====================================
# Named bundles of config values. Pick with `--preset <name>`. You can also
# override any individual flag from the command line.

PRESETS: dict[str, Config] = {
    # Final model: hybrid 1D CNN with engineered-feature side branch.
    # 112K parameters, 92.4% accuracy on participant-disjoint test split.
    "phase_a": Config(
        name="phase_a",
        model="hybrid",
        width_multiplier=1.5,
        depth=2,
        dropout=0.25,
        loss="focal",
        focal_gamma=2.0,
        class_weight="inverse",
        epochs=150,
        batch_size=64,
        learning_rate=3e-3,
        schedule="cosine_warmup",
        warmup_frac=0.05,
        early_stopping_patience=25,
        augment_shift=True,  shift_max=40,
        augment_scale=True,  scale_min=0.80, scale_max=1.20,
        augment_noise=True,  noise_std=0.05,
        augment_time_warp=True, warp_strength=0.10,
        augment_mixup=False,
    ),
}


def get_preset(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name}. available: {list(PRESETS)}")
    # return a copy so overrides don't mutate the preset
    return Config(**asdict(PRESETS[name]))
