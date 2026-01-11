#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/models/model.py

Causal Track 1 model (BIG / SOTA-oriented):
- Backbone: Dense pre-net + 3 blocuri hibride GRU + TCN (dilated Conv1D) + 1 bloc de self-attention cauzal.
- Stage 1: predict Δlog-Mel residual (tanh-scaled) so the net can both attenuate *and* amplify.
- Stage 2: multi-stage deep-filter proxy over Mel bands, gated by voicing/aux features.

I/O (matches core/train.py):
    get_model(input_dim: int) -> tf.keras.Model
    inputs:  [ (B, T, D), (B, T, 1) ]   # features and time mask (for sample weighting)
    output:  (B, T, 48)                 # predicted clean log-Mel (z-normalized)
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L

# ----------------------------- Small helpers --------------------------------- #

def _layer_norm(x, name: str):
    """LayerNorm with stable epsilon and explicit name."""
    return L.LayerNormalization(epsilon=1e-5, name=name)(x)


def _causal_depthwise_conv_over_bands(x_mel: tf.Tensor, kernel_size: int, dilation: int,
                                      name: str) -> tf.Tensor:
    """
    Depthwise (grouped) Conv1D over time, per Mel band (strictly causal).
    x_mel: [B, T, M] where M = number of Mel bands (48).
    """
    channels = int(x_mel.shape[-1])
    return L.Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation,
        groups=channels,          # per-band filtering
        use_bias=False,
        name=name,
    )(x_mel)


def _make_voicing_gate(backbone_h: tf.Tensor, aux_feats: tf.Tensor, name: str) -> tf.Tensor:
    """[B,T,48] gate in [0,1] from backbone state + auxiliary features."""
    # Aux branch (f0, vprob, ceps)
    g_from_aux = L.Dense(64, activation="swish", name=f"{name}_aux_proj")(aux_feats)
    g_from_aux = L.Dense(48, activation="sigmoid", name=f"{name}_aux_dense")(g_from_aux)

    # Backbone branch (temporal representation)
    g_from_h = L.Dense(64, activation="swish", name=f"{name}_h_proj")(backbone_h)
    g_from_h = L.Dense(48, activation="sigmoid", name=f"{name}_h_dense")(g_from_h)

    g = L.Multiply(name=f"{name}_mul")([g_from_aux, g_from_h])
    return g


def _tcn_block(x: tf.Tensor, channels: int, kernel_size: int, dilation: int,
               dropout: float, name: str) -> tf.Tensor:
    """
    Temporal ConvNet-style block (causal, dilated) with residual connection.
    x: [B, T, C], returns [B, T, C] with same number of channels.
    """
    h = L.Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation,
        activation="swish",
        name=f"{name}_conv1",
    )(x)
    h = _layer_norm(h, f"{name}_ln1")
    if dropout and dropout > 0:
        h = L.Dropout(dropout, name=f"{name}_do1")(h)

    h = L.Conv1D(
        filters=channels,
        kernel_size=1,
        padding="causal",
        activation=None,
        name=f"{name}_conv2",
    )(h)
    h = _layer_norm(h, f"{name}_ln2")
    if dropout and dropout > 0:
        h = L.Dropout(dropout, name=f"{name}_do2")(h)

    # Residual connection
    if int(x.shape[-1]) != channels:
        # Project x to the same dim if needed
        x_proj = L.Conv1D(channels, kernel_size=1, padding="causal",
                          activation=None, name=f"{name}_proj")(x)
    else:
        x_proj = x
    out = L.Add(name=f"{name}_res")([x_proj, h])
    out = L.Activation("swish", name=f"{name}_act")(out)
    return out


class _LatentAttention(L.Layer):
    """Perceiver-style latent attention: latents attend to input, then input attends to latents."""
    def __init__(self, num_latents: int, dim: int, num_heads: int, dropout: float, name: str):
        super().__init__(name=name)
        self.num_latents = int(num_latents)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.cross_in = L.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout,
            name=f"{name}_cross_in",
        )
        self.cross_out = L.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout,
            name=f"{name}_cross_out",
        )
        self.ln_in = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln_in")
        self.ln_out = L.LayerNormalization(epsilon=1e-5, name=f"{name}_ln_out")

    def build(self, input_shape):
        self.latents = self.add_weight(
            name="latents",
            shape=(self.num_latents, self.dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [B, T, D]
        bsz = tf.shape(x)[0]
        latents = tf.expand_dims(self.latents, axis=0)           # [1, L, D]
        latents = tf.repeat(latents, repeats=bsz, axis=0)        # [B, L, D]

        # Latents attend to input
        lat = self.cross_in(latents, x, x)
        lat = self.ln_in(lat + latents)

        # Input attends to latents
        out = self.cross_out(x, lat, lat)
        out = self.ln_out(out + x)
        return out


# ------------------------------- Model --------------------------------------- #

def get_model(input_dim: int,
              hidden: int = 256,
              df_kernel: int = 7,
              dropout: float = 0.1) -> tf.keras.Model:
    """
    Build the Track 1 BIG model.

    - input_dim: D (e.g., 50 for 48 Mel + F0 + vprob [+ cepstra])
    - hidden:    main model width (256 recommended for SOTA push; 192–320 reasonable)
    - df_kernel: deep-filter kernel size (5–11 recommended); causal
    - dropout:   dropout after major blocks
    """
    assert input_dim >= 50, "Expected at least 48 Mel + F0 + vprob"

    # Inputs
    xin  = L.Input(shape=(None, input_dim), name="features")   # [B, T, D]
    xmsk = L.Input(shape=(None, 1),         name="time_mask")  # [B, T, 1] (sample weights / unused in graph)

    # Split features: [mel_log(48), aux(...)]
    mel_log = L.Lambda(lambda z: z[:, :, :48], name="slice_mel")(xin)   # [B, T, 48]
    aux     = L.Lambda(lambda z: z[:, :, 48:], name="slice_aux")(xin)   # [B, T, D-48]

    # -------------------------- Mel multi-branch front-end -------------------- #
    # Multi-scale causal convs over mel to enrich short/medium/long contexts.
    m1 = L.Conv1D(32, kernel_size=3, padding="causal", activation="swish", name="mel_br_k3")(mel_log)
    m2 = L.Conv1D(32, kernel_size=7, padding="causal", activation="swish", name="mel_br_k7")(mel_log)
    m3 = L.Conv1D(32, kernel_size=15, padding="causal", activation="swish", name="mel_br_k15")(mel_log)
    mel_multi = L.Concatenate(name="mel_multi")([mel_log, m1, m2, m3])
    mel_multi = _layer_norm(mel_multi, "mel_multi_ln")

    # -------------------------- Pre-net (input projection) -------------------- #
    x_in = L.Concatenate(name="in_concat")([mel_multi, aux])
    h = L.Dense(hidden, activation="swish", name="in_proj")(x_in)       # [B, T, H]
    h = _layer_norm(h, "in_ln")

    # -------------------------- Hybrid GRU + TCN backbone --------------------- #
    # 3 blocuri: GRU → LN/Dropout → TCN (dilated) → residual cu inputul blocului
    for i in range(3):
        # GRU (causal, unidirectional)
        h_gru = L.GRU(
            hidden,
            return_sequences=True,
            name=f"gru{i+1}"
        )(h)
        h_gru = _layer_norm(h_gru, f"gru{i+1}_ln")
        if dropout and dropout > 0:
            h_gru = L.Dropout(dropout, name=f"gru{i+1}_do")(h_gru)

        # TCN block (dilated Conv1D)
        dilation = 2 ** i
        h_tcn = _tcn_block(
            h_gru,
            channels=hidden,
            kernel_size=5,
            dilation=dilation,
            dropout=dropout,
            name=f"tcn{i+1}"
        )

        # Residual block output
        h = L.Add(name=f"block{i+1}_out")([h, h_tcn])

    # -------------------------- Latent attention ------------------------------ #
    # Perceiver-style: latents <-> input cross-attention (no causal mask needed).
    lat_attn = _LatentAttention(
        num_latents=16,
        dim=hidden,
        num_heads=4,
        dropout=dropout,
        name="latent_attn",
    )(h)
    if dropout and dropout > 0:
        lat_attn = L.Dropout(dropout, name="attn_do")(lat_attn)
    h = L.Activation("swish", name="attn_act")(lat_attn)

    # ---------------- Stage 1: Δlog-Mel residual (boost + cut) ---------------- #
    # Δ in approx ±1.5 log-power (~±6.5 dB).
    delta_hidden = L.Dense(hidden, activation="swish", name="delta_hid")(h)
    delta = L.Dense(48, activation="tanh", name="delta_logmel")(delta_hidden)  # [-1,1]
    delta = L.Lambda(lambda d: d * 1.5, name="delta_scale")(delta)

    masked_logmel = L.Add(name="masked_logmel")([mel_log, delta])  # [B, T, 48]

    # ---------------- Stage 2: Multi-stage deep filtering -------------------- #
    # 3 stacked depthwise Conv1D blocks over Mel bands, with residuals.
    df = masked_logmel
    for j in range(3):
        df_block = _causal_depthwise_conv_over_bands(
            df, kernel_size=df_kernel, dilation=(2 ** j), name=f"df_dw{j+1}"
        )
        df_block = _layer_norm(df_block, f"df_ln{j+1}")
        if dropout and dropout > 0:
            df_block = L.Dropout(dropout, name=f"df_do{j+1}")(df_block)
        df = L.Add(name=f"df_res{j+1}")([df, df_block])

    # Compute residual versus the masked_logmel baseline
    df_residual = L.Subtract(name="df_residual")([df, masked_logmel])

    # Voicing gate from backbone state + aux features
    v_gate = _make_voicing_gate(h, aux, name="v_gate")  # [B, T, 48]
    df_residual = L.Multiply(name="df_residual_gated")([df_residual, v_gate])

    logmel_hat = L.Add(name="logmel_hat")([masked_logmel, df_residual])  # [B, T, 48]

    # -------------------------- Output projection ----------------------------- #
    logmel_hat = _layer_norm(logmel_hat, "out_ln")
    logmel_hat = L.Dense(48, activation=None, name="out_proj")(logmel_hat)
    # Keep time_mask connected to the graph (no-op) to satisfy Functional API.
    logmel_hat = L.Lambda(
        lambda t: t[0] + 0.0 * t[1],
        name="connect_time_mask"
    )([logmel_hat, xmsk])

    model = tf.keras.Model(
        inputs=[xin, xmsk],
        outputs=logmel_hat,
        name="track1_big_mask_df"
    )

    return model
