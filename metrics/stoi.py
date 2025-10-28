from __future__ import annotations
import numpy as np

from metrics_utils import to_mono, align, finite_or_default, clamp, LOG

try:
    from pystoi.stoi import stoi as _stoi
except Exception as e:
    _stoi = None
    _IMPORT_ERR = e


def stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int, extended: bool = False) -> float:
    """
    Compute STOI (or eSTOI if extended=True).
    Signals must be mono, same length, float. Raises on error.
    """
    if _stoi is None:
        raise ImportError(
            "The 'pystoi' package is required for STOI computation."
        ) from _IMPORT_ERR
    if not isinstance(ref, np.ndarray) or not isinstance(deg, np.ndarray):
        raise TypeError("ref/deg must be numpy arrays")
    if sr <= 0:
        raise ValueError("sr must be positive")
    ref = ref.astype(np.float32).ravel()
    deg = deg.astype(np.float32).ravel()
    ref, deg = align(ref, deg)
    if len(ref) == 0:
        raise ValueError("Empty input")
    return float(_stoi(ref, deg, sr, extended=extended))


from resample_audio import resample_audio 

def stoi_score_safe(ref, deg, sr, extended: bool = False, default: float = 0.0, auto_resample: bool = False) -> float:
    try:
        ref = to_mono(ref); deg = to_mono(deg)
        ref, deg = align(ref, deg)
        if len(ref) == 0:
            raise ValueError("Empty input")
        target_sr = 10000 if not extended else sr  # classic STOI commonly uses 10k
        if auto_resample and not extended and sr != 10000:
            ref = resample_audio(ref, sr, 10000)
            deg = resample_audio(deg, sr, 10000)
            sr = 10000
        val = stoi_score(ref, deg, sr, extended=extended)
        return clamp(finite_or_default(val, default, "STOI"), 0.0, 1.0)
    except Exception as e:
        LOG.warning("STOI failed (sr=%s): %s. Using default=%s.", sr, e, default)
        return float(default)
