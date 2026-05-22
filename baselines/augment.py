"""Label-preserving augmentations for PPG windows.

The point of augmentation is to multiply the effective dataset size by showing
the model many *variations* of each window. Variations the model learns to
ignore = robustness. Cardiac diagnosis must be invariant to small time shifts,
amplitude differences, sensor noise, and minor speed variation.
"""
from __future__ import annotations

import tensorflow as tf


def make_augmenter(
    *,
    do_shift: bool = True,
    shift_max: int = 32,
    do_scale: bool = True,
    scale_range: tuple = (0.85, 1.15),
    do_noise: bool = True,
    noise_std: float = 0.05,
    do_time_warp: bool = False,
    warp_strength: float = 0.1,
):
    """Build an augmentation function based on enabled flags.

    Args:
        do_shift: random circular shift in time (mimics arbitrary window start).
        shift_max: max shift in samples (32 samples = 0.64 s at 50 Hz).
        do_scale: random amplitude scaling (mimics sensor sensitivity drift).
        scale_range: low/high scale factors. (0.85, 1.15) = ±15%.
        do_noise: add Gaussian noise (mimics electrical / quantization noise).
        noise_std: noise standard deviation in normalized signal units.
        do_time_warp: stretch/squeeze the signal in time (mimics HR variability).
        warp_strength: max relative warp. 0.1 = ±10% speed change.

    Returns:
        A `tf.function` `(x, y) -> (x_aug, y)` for use in `tf.data.Dataset.map`.

    Notes:
        * All ops are label-preserving — they do not change whether the window
          is cardiac, occlusion, or off-body.
        * Mixup is handled separately by the trainer because it operates across
          two samples, not on a single sample.
    """
    scale_lo, scale_hi = scale_range

    @tf.function
    def aug(x, y):
        if do_shift:
            shift = tf.random.uniform([], -shift_max, shift_max + 1, dtype=tf.int32)
            x = tf.roll(x, shift=shift, axis=0)
        if do_scale:
            x = x * tf.random.uniform([], scale_lo, scale_hi)
        if do_noise:
            x = x + tf.random.normal(tf.shape(x), stddev=noise_std)
        if do_time_warp:
            # Resample to a stretched/squeezed length, then re-resample back to 512.
            n = tf.shape(x)[0]
            factor = tf.random.uniform([], 1.0 - warp_strength, 1.0 + warp_strength)
            new_n = tf.cast(tf.cast(n, tf.float32) * factor, tf.int32)
            # Linear interp via tf.image.resize on a (length, 1, 1) tensor.
            x_img = tf.reshape(x, (1, n, 1, 1))
            x_warp = tf.image.resize(x_img, [new_n, 1], method="bilinear")
            x_back = tf.image.resize(x_warp, [n, 1], method="bilinear")
            x = tf.reshape(x_back, tf.shape(x))
        return x, y

    return aug


def apply_mixup(x_batch, y_batch_onehot, alpha: float = 0.2):
    """Mix pairs of samples in a batch: new_x = a*x_i + (1-a)*x_j.

    Args:
        x_batch: (B, L, 1) signal batch.
        y_batch_onehot: (B, C) one-hot label batch.
        alpha: Beta distribution parameter. Higher = more mixing.
            * 0.0 → mixup disabled (identity)
            * 0.1–0.2 → mild mixup (typical for time-series)
            * 0.4 → strong mixup (often used on images)
            * 1.0 → very aggressive — usually hurts

    Returns:
        Mixed (x, y) batch. Labels are soft (no longer one-hot).

    Notes:
        * Use with cross-entropy or KL-loss, NOT sparse cross-entropy.
        * Especially helpful when the model overfits to per-patient morphology.
    """
    batch_size = tf.shape(x_batch)[0]
    lam = tf.random.gamma([batch_size, 1, 1], alpha) / (
        tf.random.gamma([batch_size, 1, 1], alpha) + 1e-8 +
        tf.random.gamma([batch_size, 1, 1], alpha))
    lam = tf.clip_by_value(lam, 0.0, 1.0)
    idx = tf.random.shuffle(tf.range(batch_size))
    x_mix = lam * x_batch + (1.0 - lam) * tf.gather(x_batch, idx)
    lam_y = tf.reshape(lam, [batch_size, 1])
    y_mix = lam_y * y_batch_onehot + (1.0 - lam_y) * tf.gather(y_batch_onehot, idx)
    return x_mix, y_mix
