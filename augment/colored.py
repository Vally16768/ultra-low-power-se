# augment/colored.py
import numpy as np
from .base import register


def _colored_noise(N, color, rng: np.random.Generator):
    alpha = {"white": 0, "pink": 1, "brown": 2}[color]
    X = np.fft.rfft(rng.standard_normal(N))
    freqs = np.fft.rfftfreq(N)
    w = (freqs + 1e-6) ** (-alpha / 2)
    return np.fft.irfft(X * w, n=N).astype(np.float32)


@register("add_colored_noise")
def add_colored_noise(x: np.ndarray, sr: int, rng: np.random.Generator, color: str = "pink", snr_db: float = 5.0) -> np.ndarray:
    n = _colored_noise(len(x), color, rng)
    ps = (x**2).mean() + 1e-12
    pn_target = ps / (10 ** (snr_db / 10))
    n = n / (n.std() + 1e-12) * np.sqrt(pn_target)
    return x + n
