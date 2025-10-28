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


def stoi_score_safe(ref, deg, sr, extended: bool = False, default: float = 0.0) -> float:
    """
    Training-safe STOI/eSTOI:
      * folds to mono, aligns
      * clamps to [0,1]
      * never returns NaN/Inf; returns default on failure
    """
    try:
        ref = to_mono(ref)
        deg = to_mono(deg)
        ref, deg = align(ref, deg)
        if len(ref) == 0:
            raise ValueError("Empty input")
        val = stoi_score(ref, deg, sr, extended=extended)
        return clamp(finite_or_default(val, default, "STOI"), 0.0, 1.0)
    except Exception as e:
        LOG.warning("STOI failed (sr=%s): %s. Using default=%s.", sr, e, default)
        return float(default)
