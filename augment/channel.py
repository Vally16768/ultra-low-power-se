# augment/channel.py
import numpy as np
from .base import register


@register("eq_tilt")
def eq_tilt(x: np.ndarray, sr: int, rng: np.random.Generator, db_per_oct: float | None = None) -> np.ndarray:
    tilt = db_per_oct if db_per_oct is not None else float(rng.uniform(-6, 6))
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x)) * sr
    octs = np.log2(np.maximum(f, 1.0) / 1000.0)  # referință 1 kHz
    g = 10 ** ((tilt / 20.0) * octs)
    Y = X * g
    y = np.fft.irfft(Y, n=len(x)).astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-6
    return y


@register("soft_clip")
def soft_clip(x: np.ndarray, sr: int, rng: np.random.Generator, drive: float = 0.8):
    return np.tanh(drive * x).astype(np.float32)
