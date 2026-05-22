"""Loss functions for imbalanced classification.

Cross-entropy is the default. Focal loss biases learning toward hard examples,
which matters when one class (occlusion) is both rare and clinically critical.
"""
from __future__ import annotations

import tensorflow as tf


def focal_loss(gamma: float = 2.0, alpha=None):
    """Categorical focal loss.

    Args:
        gamma: focus parameter. Higher = more emphasis on hard examples.
            * gamma=0 → equivalent to plain cross-entropy
            * gamma=1 → mild emphasis on hard examples
            * gamma=2 → typical setting (Lin et al. 2017)
            * gamma=3+ → very aggressive, can starve easy classes
        alpha: per-class weighting tensor of shape (n_classes,). If None, no
            extra weighting. Use this OR `class_weight` in `model.fit`, not both.

    Returns:
        A loss function `(y_true_sparse, y_pred_prob) -> scalar`.

    When to use: occlusion is 10% of the data and we already over-predict it
    with plain CE + class weights (recall good, precision bad). Focal loss
    sharpens the model's focus on examples it currently gets wrong, which
    often improves precision without sacrificing recall.
    """
    if alpha is not None:
        alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        # y_true is sparse (integer labels); y_pred is softmax probabilities.
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_oh = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        ce = -y_true_oh * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1, keepdims=True)
        modulating = tf.pow(1.0 - p_t, gamma)
        fl = modulating * ce
        if alpha is not None:
            fl = fl * alpha
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    return loss


def make_loss(name: str, gamma: float = 2.0):
    """Factory used by the trainer based on config.loss."""
    if name == "cross_entropy":
        return "sparse_categorical_crossentropy"
    if name == "focal":
        return focal_loss(gamma=gamma)
    raise ValueError(f"unknown loss: {name}")
