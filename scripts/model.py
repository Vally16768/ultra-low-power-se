from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L, models

def build_keras(input_dim: int, output_dim: int, lr=1e-3, dropout=0.2):
    inp = L.Input(shape=(input_dim,), name="feat")
    x = L.LayerNormalization()(inp)
    x = L.Dense(512, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(output_dim, activation="sigmoid", name="mask")(x)  # 0..1
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return m
