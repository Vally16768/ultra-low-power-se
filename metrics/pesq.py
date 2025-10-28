from __future__ import annotations
import sys
import importlib
import numpy as np

from metrics_utils import to_mono, align, finite_or_default, clamp, LOG

# Try to obtain a real PESQ backend without shadowing ourselves.
_pesq_func = None
_IMPORT_ERR = None

def _load_pesq_backend():
    global _pesq_func, _IMPORT_ERR
    try:
        pkg = importlib.import_module("pesq")
        # If pkg is THIS module (shadowed), force an ImportError to try fallback
        if pkg is sys.modules.get(__name__):
            raise ImportError("Local module 'pesq.py' shadows the third-party 'pesq' package.")
        if hasattr(pkg, "pesq"):
            _pesq_func = pkg.pesq
            return
    except Exception as e:
        _IMPORT_ERR = e

    # Fallback: pypesq (wideband only; not bit-exact ITU but better than nothing)
    try:
        import pypesq
        def _pypesq_wrapper(sr, ref, deg, mode):
            # pypesq signature: pypesq.pesq(ref, deg, sr)
            return float(pypesq.pesq(ref, deg, sr))
        _pesq_func = _pypesq_wrapper
        return
    except Exception as e2:
        _IMPORT_ERR = Exception(f"{_IMPORT_ERR}; fallback pypesq failed: {e2}")

_load_pesq_backend()


def _auto_mode(sr: int) -> str:
    # ITU PESQ supports nb(8k) and wb(16k). We'll use nb at 8k, wb otherwise.
    return "nb" if sr <= 8000 else "wb"


def _resample_linear(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float32)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    n_out = int(np.floor(len(x) * (fs_out / fs_in)))
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)


def pesq_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    Compute PESQ between reference (clean) and degraded/enhanced.
    - sr: 8000→'nb', >=16000→'wb' (auto)
    Signals must be mono, same length, float. This function raises on error.
    """
    if _pesq_func is None:
        raise ImportError(
            "No PESQ backend available. Install 'pesq' (preferred) or 'pypesq'."
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

    mode = _auto_mode(sr)
    return float(_pesq_func(sr, ref, deg, mode))


def pesq_score_safe(
    ref, deg, sr, default: float = 1.5, auto_resample: bool = True
) -> float:
    """
    Training-safe PESQ:
      * folds to mono, aligns, optional auto-resample to 8k/16k
      * clamps to legal range [-0.5, 4.5]
      * never returns NaN/Inf; returns default on failure
    """
    try:
        ref = to_mono(ref)
        deg = to_mono(deg)
        ref, deg = align(ref, deg)
        if len(ref) == 0:
            raise ValueError("Empty input")

        # Resample to supported rates for the standard backend
        target_sr = 16000 if sr >= 12000 else 8000
        if auto_resample and sr not in (8000, 16000):
            ref = _resample_linear(ref, sr, target_sr)
            deg = _resample_linear(deg, sr, target_sr)
            sr = target_sr

        val = pesq_score(ref, deg, sr)
        return clamp(finite_or_default(val, default, "PESQ"), -0.5, 4.5)
    except Exception as e:
        LOG.warning("PESQ failed (sr=%s): %s. Using default=%s.", sr, e, default)
        return float(default)
