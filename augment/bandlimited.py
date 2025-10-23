# augment/bandlimited.py
import numpy as np
from scipy.signal import butter, lfilter
from .base import register


@register("bandlimit_noise")
def bandlimit_noise(x: np.ndarray, sr: int, rng: np.random.Generator, lo: int | None = None, hi: int | None = None, snr_db: float = 5.0) -> np.ndarray:
    lo = lo or int(rng.uniform(100, 800))
    hi = hi or int(rng.uniform(2000, 6000))
    w = np.random.standard_normal(len(x)).astype(np.float32)
    b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    n = lfilter(b, a, w)
    ps = (x**2).mean() + 1e-12
    pn_target = ps / (10 ** (snr_db / 10))
    n = n / (np.std(n) + 1e-12) * np.sqrt(pn_target)
    return x + n
