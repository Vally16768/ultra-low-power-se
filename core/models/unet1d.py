from __future__ import annotations
from typing import Optional
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x, ch, k=9, s=1):
    x = layers.Conv1D(ch, k, s, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def down_block(x, ch):
    c1 = conv_block(x, ch)
    c2 = conv_block(c1, ch)
    p  = layers.AveragePooling1D(2)(c2)
    return c2, p

def up_block(x, skip, ch):
    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, ch)
    x = conv_block(x, ch)
    return x

def build_unet1d(input_len: Optional[int], base_ch: int = 64) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="noisy_in")
    s1, p1 = down_block(inp, base_ch)
    s2, p2 = down_block(p1,  base_ch * 2)
    s3, p3 = down_block(p2,  base_ch * 4)
    b      = conv_block(p3,  base_ch * 8)
    u3 = up_block(b,  s3, base_ch * 4)
    u2 = up_block(u3, s2, base_ch * 2)
    u1 = up_block(u2, s1, base_ch)
    out = layers.Conv1D(1, 1, padding="same", name="enhanced")(u1)
    return keras.Model(inp, out, name="UNet1D_SE")
