# augment/bursts.py
import numpy as np
from .base import register


@register("add_bursts")
def add_bursts(x: np.ndarray, sr: int, rng: np.random.Generator, n_bursts: int = 3, dur_ms: int = 60) -> np.ndarray:
    y = x.copy()
    L = len(x)
    B = int(dur_ms * sr / 1000)
    for _ in range(n_bursts):
        s = int(rng.integers(0, max(1, L - B)))
        y[s : s + B] += 0.2 * rng.standard_normal(B).astype(np.float32)
    return np.clip(y, -1, 1)
