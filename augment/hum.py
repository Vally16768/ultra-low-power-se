# augment/hum.py
import numpy as np
from .base import register


@register("add_hum")
def add_hum(x: np.ndarray, sr: int, rng: np.random.Generator, base_hz: float | None = None, snr_db: float = 10.0) -> np.ndarray:
    f0 = base_hz if base_hz else rng.choice([50.0, 60.0])
    t = np.arange(len(x)) / sr
    n = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    ps = (x**2).mean() + 1e-12
    pn_target = ps / (10 ** (snr_db / 10))
    n = n / (n.std() + 1e-12) * np.sqrt(pn_target)
    return x + n.astype(np.float32)
