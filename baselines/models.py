"""Model architectures for the cardiac classifier.

Each builder takes the config and returns a Keras model. Models that need
features as a second input return models with two inputs. The trainer
checks `model.input_names` to know whether to provide features.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------- building blocks ------------------------------

def _conv_block(x, filters: int, kernel: int, stride: int = 2,
                use_bn: bool = True, dropout: float = 0.0):
    x = layers.Conv1D(filters, kernel, strides=stride, padding="same",
                      use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def _residual_block(x, filters: int, kernel: int = 3, dropout: float = 0.0):
    """Two convs with a skip connection. Lets gradient flow through deeper nets."""
    shortcut = x
    in_filters = x.shape[-1]
    x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if in_filters != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# ------------------------------- architectures ------------------------------

def build_tiny_cnn(cfg) -> tf.keras.Model:
    """9.4K-param baseline (our original 'basic' model)."""
    inp = layers.Input(shape=(cfg.input_len, 1), name="signal")
    x = _conv_block(inp, 16, 7)
    x = _conv_block(x, 32, 5)
    x = _conv_block(x, 64, 3)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(cfg.n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="tiny_cnn")


def build_improved_cnn(cfg) -> tf.keras.Model:
    """24K-param 1D CNN with light dropout (the 85% model)."""
    inp = layers.Input(shape=(cfg.input_len, 1), name="signal")
    x = _conv_block(inp, 20, 9)
    x = _conv_block(x, 32, 7, dropout=0.1)
    x = _conv_block(x, 48, 5, dropout=0.1)
    x = _conv_block(x, 64, 3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(cfg.n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="improved_cnn")


def build_resnet1d(cfg) -> tf.keras.Model:
    """Bigger residual 1D CNN (~150K-300K params depending on width).

    Useful when accuracy matters more than chip size. Residual connections
    let it go deeper without vanishing gradients.
    """
    w = cfg.width_multiplier
    d = cfg.depth
    inp = layers.Input(shape=(cfg.input_len, 1), name="signal")
    x = _conv_block(inp, int(32 * w), 9)
    x = _conv_block(x, int(64 * w), 7)
    for _ in range(d):
        x = _residual_block(x, int(64 * w), kernel=3, dropout=cfg.dropout)
    x = _conv_block(x, int(128 * w), 3)
    for _ in range(d):
        x = _residual_block(x, int(128 * w), kernel=3, dropout=cfg.dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(cfg.n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="resnet1d")


def build_hybrid(cfg) -> tf.keras.Model:
    """CNN on raw signal + small MLP on engineered features, fused.

    Two-headed network. The CNN learns waveform morphology; the MLP injects
    HR-band statistics the CNN would otherwise need lots of data to learn.
    """
    sig = layers.Input(shape=(cfg.input_len, 1), name="signal")
    feats = layers.Input(shape=(cfg.n_features,), name="features")
    w = cfg.width_multiplier

    # signal branch
    s = _conv_block(sig, int(20 * w), 9)
    s = _conv_block(s, int(32 * w), 7, dropout=cfg.dropout * 0.5)
    s = _conv_block(s, int(48 * w), 5, dropout=cfg.dropout * 0.5)
    s = _conv_block(s, int(64 * w), 3)
    if cfg.depth > 0:
        s = _residual_block(s, int(64 * w), kernel=3, dropout=cfg.dropout * 0.5)
    s = layers.GlobalAveragePooling1D()(s)

    # feature branch
    f = layers.Dense(32, activation="relu")(feats)
    f = layers.Dropout(cfg.dropout)(f)
    f = layers.Dense(16, activation="relu")(f)

    # fuse
    x = layers.Concatenate()([s, f])
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(cfg.n_classes, activation="softmax")(x)
    return models.Model([sig, feats], out, name="hybrid_cnn")


# ------------------------------- registry -----------------------------------

BUILDERS = {
    "tiny_cnn":     build_tiny_cnn,
    "improved_cnn": build_improved_cnn,
    "resnet1d":     build_resnet1d,
    "hybrid":       build_hybrid,
}


def build(cfg):
    if cfg.model not in BUILDERS:
        raise ValueError(f"unknown model: {cfg.model}. options: {list(BUILDERS)}")
    return BUILDERS[cfg.model](cfg)


def needs_features(model: tf.keras.Model) -> bool:
    return any(inp.name.startswith("features") for inp in model.inputs)
