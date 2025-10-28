# core/data/audio_utils.py
from __future__ import annotations
import numpy as np
import soundfile as sf

# Use unified, audio-safe resampler
from resample_audio import resample_audio


def _sanitize_wav(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    # keep extreme values bounded and finite; avoid silent clipping surprises
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(x, -1.0, 1.0)


def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    """
    Load audio from `path`, fold to mono, resample to `target_sr` using resample_audio(),
    and sanitize to a finite, float32, [-1,1] array.
    """
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=-1)
    x = x.astype(np.float32, copy=False)
    if sr != target_sr:
        x = resample_audio(x, sr, target_sr)
    return _sanitize_wav(x)


def pad_or_random_crop(x: np.ndarray, segment_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Ensure a fixed-length segment by random crop or zero-pad (left-aligned).
    """
    x = _sanitize_wav(x)
    L = int(x.shape[0])
    if L == segment_len:
        return x
    if L > segment_len:
        start = int(rng.integers(0, L - segment_len + 1))
        return x[start : start + segment_len]
    out = np.zeros((segment_len,), dtype=np.float32)
    out[:L] = x
    return out
