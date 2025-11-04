#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/models/model.py

Best-quality Track 1 architecture (paper-derived, causal):
- Stage 1: per-band mask (sigmoid gains) via 2×GRU backbone
- Stage 2: deep filtering proxy via grouped (depthwise) causal Conv1D over Mel bands,
          gated by voicing/periodicity features predicted from the backbone.

I/O (matches core/train.py):
    get_model(input_dim: int) -> tf.keras.Model
    inputs:  [ (B, T, D), (B, T, 1) ]   # features and time mask (for sample weighting)
    output:  (B, T, 48)                 # predicted clean log-Mel
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L


# ----------------------------- Small helpers --------------------------------- #

def _layer_norm(x, name):
    return L.LayerNormalization(epsilon=1e-5, name=name)(x)

def _causal_depthwise_conv_over_bands(x_mel: tf.Tensor, kernel_size: int, name: str) -> tf.Tensor:
    """Depthwise (grouped) Conv1D over time, per Mel band (strictly causal)."""
    channels = int(x_mel.shape[-1])
    return L.Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        groups=channels,
        use_bias=False,
        name=name,
    )(x_mel)

def _make_voicing_gate(backbone_h: tf.Tensor, aux_feats: tf.Tensor, name: str) -> tf.Tensor:
    """[B,T,48] gate in [0,1] from backbone state + auxiliary features."""
    g_from_aux = L.Dense(48, activation="sigmoid", name=f"{name}_aux_dense")(aux_feats)
    g_from_h   = L.Dense(48, activation="sigmoid", name=f"{name}_h_dense")(backbone_h)
    return L.Multiply(name=f"{name}_mul")([g_from_aux, g_from_h])


# ------------------------------- Model --------------------------------------- #

def get_model(input_dim: int,
              hidden: int = 128,
              df_kernel: int = 7,
              dropout: float = 0.05) -> tf.keras.Model:
    """
    Build the Track 1 model.
    - input_dim: D (e.g., 50 for 48 Mel + F0 + vprob [+ cepstra])
    - hidden:    GRU hidden units (96–128 recommended)
    - df_kernel: deep-filter kernel size (5–9 recommended); causal
    - dropout:   small dropout after GRUs
    """
    assert input_dim >= 50, "Expected at least 48 Mel + F0 + vprob"

    xin  = L.Input(shape=(None, input_dim), name="features")   # [B, T, D]
    xmsk = L.Input(shape=(None, 1),         name="time_mask")  # [B, T, 1] (sample weights)

    # Split features
    mel_log = L.Lambda(lambda z: z[:, :, :48], name="slice_mel")(xin)   # [B, T, 48]
    aux     = L.Lambda(lambda z: z[:, :, 48:], name="slice_aux")(xin)   # [B, T, D-48]

    # ---------------- Stage 1: Mask head (coarse suppression) ---------------- #
    h = L.GRU(hidden, return_sequences=True, name="gru1")(xin)
    h = _layer_norm(h, "ln1")
    if dropout and dropout > 0: h = L.Dropout(dropout, name="do1")(h)

    h = L.GRU(hidden, return_sequences=True, name="gru2")(h)
    h = _layer_norm(h, "ln2")
    if dropout and dropout > 0: h = L.Dropout(dropout, name="do2")(h)

    # Bias-init to logit(0.8) so gains start near 0.8 (less early over-suppression)
    gains = L.Dense(
        48,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(1.386294),  # log(0.8/0.2)
        name="gains",
    )(h)  # [B, T, 48] in [0,1]

    # Apply mask in log domain: log(|Y|^2) = log(|X|^2) + log(gain^2)
    def _log_gain_layer(g):
        eps = tf.constant(1e-6, dtype=g.dtype)
        return tf.math.log(tf.clip_by_value(tf.square(g), eps, 1.0))
    log_gain2 = L.Lambda(_log_gain_layer, name="log_gain2")(gains)  # [B, T, 48]

    masked_logmel = L.Add(name="masked_logmel")([mel_log, log_gain2])  # [B, T, 48]

    # ----------- Stage 2: Deep filtering (fine cleanup of voiced speech) ------ #
    df_residual = _causal_depthwise_conv_over_bands(masked_logmel, kernel_size=df_kernel, name="df_depthwise")
    df_residual = _layer_norm(df_residual, "df_ln")

    v_gate = _make_voicing_gate(h, aux, name="v_gate")  # [B, T, 48]
    df_residual = L.Multiply(name="df_gated")([df_residual, v_gate])

    logmel_hat = L.Add(name="logmel_hat")([masked_logmel, df_residual])  # [B, T, 48]
    logmel_hat = _layer_norm(logmel_hat, "out_ln")
    logmel_hat = L.Dense(48, activation=None, name="out_proj")(logmel_hat)

    model = tf.keras.Model(inputs=[xin, xmsk], outputs=logmel_hat, name="track1_mask_df")
    model.add_metric(L.Lambda(lambda g: tf.reduce_mean(g))(gains), name="gain_mean", aggregation="mean")
    return model
