# augment/reverb.py
import numpy as np
from .base import register


@register("reverb_toy")
def reverb_toy(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    delays = [int(sr * d) for d in rng.uniform(0.008, 0.045, size=3)]
    gains = list(rng.uniform(0.2, 0.6, size=3))
    for d, g in zip(delays, gains):
        pad = np.zeros_like(x)
        pad[d:] = x[:-d]
        y += g * pad
    return y
