# augment/babble.py
import numpy as np, random, soundfile as sf
from .base import register


@register("add_babble")
def add_babble(x: np.ndarray, sr: int, rng: np.random.Generator, pool_paths: list[str] = None, speakers: int = 4, snr_db: float = 0.0) -> np.ndarray:
    assert pool_paths and len(pool_paths) >= speakers, "Need pool_paths for babble"
    mix = np.zeros_like(x, dtype=np.float32)
    for p in rng.choice(pool_paths, size=speakers, replace=False):
        y, fs = sf.read(p, dtype="float32", always_2d=False)
        if fs != sr:
            raise RuntimeError("Resample offline to keep module pure-numpy.")
        if y.ndim > 1:
            y = y.mean(-1)
        if len(y) < len(x):
            reps = int(np.ceil(len(x) / len(y)))
            y = np.tile(y, reps)
        y = y[: len(x)]
        mix += y / max(1, speakers)  # normalizare simplă
    # scale la SNR
    ps = (x**2).mean() + 1e-12
    pn_target = ps / (10 ** (snr_db / 10))
    mix = mix / (mix.std() + 1e-12) * np.sqrt(pn_target)
    return x + mix
