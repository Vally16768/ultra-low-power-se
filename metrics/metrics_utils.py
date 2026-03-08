from __future__ import annotations
import logging
import numpy as np
import librosa

LOG = logging.getLogger("metrics")
LOG.addHandler(logging.NullHandler())

def to_mono(x: np.ndarray) -> np.ndarray:
    """Fold to mono, flatten, float32."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    return x.astype(np.float32).ravel()

def align(a: np.ndarray, b: np.ndarray):
    """Trim both arrays to the same (min) length."""
    n = min(len(a), len(b))
    return a[:n], b[:n]

def resample_audio(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio only when needed."""
    x = to_mono(x)
    if int(orig_sr) == int(target_sr):
        return x
    return librosa.resample(x, orig_sr=int(orig_sr), target_sr=int(target_sr), res_type="kaiser_fast")

def extract_overlapped_windows(x: np.ndarray, nperseg: int, noverlap: int, window: np.ndarray | None = None) -> np.ndarray:
    """Create overlapped analysis windows, matching the pysepm conventions."""
    if nperseg <= noverlap:
        raise ValueError("nperseg must be greater than noverlap.")
    x = to_mono(x)
    if x.shape[-1] < nperseg:
        x = np.pad(x, (0, nperseg - x.shape[-1]))
    step = nperseg - noverlap
    n_frames = 1 + (x.shape[-1] - nperseg) // step
    shape = x.shape[:-1] + (n_frames, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    if window is not None:
        result = window * result
    return result.copy()

def finite_or_default(value: float, default: float, name: str):
    """Return value if finite; otherwise log + return default."""
    if value is None or not np.isfinite(value):
        LOG.warning("Metric %s returned non-finite value (%r). Using default=%r.", name, value, default)
        return float(default)
    return float(value)

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp scalar to [lo, hi]."""
    return float(max(lo, min(hi, val)))

def nan_to_num_inplace(x: np.ndarray, nan: float = 0.0):
    """Scrub NaN/Inf from waveforms (optional pre-step)."""
    if isinstance(x, np.ndarray):
        np.nan_to_num(x, copy=False, nan=nan, posinf=1e9, neginf=-1e9)
    return x

def frame_config(fs: int, frame_len: float = 0.03, overlap: float = 0.75) -> tuple[int, int]:
    """Return analysis window length and hop in samples."""
    winlength = max(1, int(round(frame_len * fs)))
    hop = max(1, int(np.floor((1.0 - overlap) * frame_len * fs)))
    return winlength, hop

def hann_window(winlength: int) -> np.ndarray:
    """Use the same Hann definition as the reference composite code."""
    idx = np.arange(1, winlength + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (winlength + 1.0)))
