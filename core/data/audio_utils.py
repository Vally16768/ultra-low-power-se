# core/data/audio_utils.py
from __future__ import annotations
import numpy as np
import soundfile as sf
import scipy.signal as sps

def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x.astype(np.float32, copy=False)
    g = np.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    y = sps.resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)

def _sanitize_wav(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(x, -1.0, 1.0)

def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = _resample(x, sr, target_sr)
    return _sanitize_wav(x)

def pad_or_random_crop(x: np.ndarray, segment_len: int, rng: np.random.Generator) -> np.ndarray:
    x = _sanitize_wav(x)
    L = x.shape[0]
    if L == segment_len:
        return x
    if L > segment_len:
        start = int(rng.integers(0, L - segment_len + 1))
        return x[start:start + segment_len]
    out = np.zeros((segment_len,), dtype=np.float32)
    out[:L] = x
    return out
