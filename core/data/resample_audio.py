# core/data/resample_audio.py
import numpy as np

try:
    from scipy.signal import resample_poly as _resample_poly
except Exception as e:
    _SCIPY_AVAILABLE = False
    _IMPORT_ERR = e
else:
    _SCIPY_AVAILABLE = True
    _IMPORT_ERR = None


def resample_audio(
    x: np.ndarray,
    sr: int,
    target_sr: int,
    *,
    axis: int = -1,
    dtype=np.float32,
    kaiser_beta: float = 8.6,
) -> np.ndarray:
    """
    High-quality, robust resampling with polyphase FIR filtering.

    Parameters
    ----------
    x : np.ndarray
        Signal array (1D or ND). The dimension to be resampled is `axis`.
    sr : int
        Original sampling rate (Hz).
    target_sr : int
        Target sampling rate (Hz).
    axis : int, default -1
        Axis along which to resample.
    dtype : numpy dtype, default np.float32
        Output dtype (and cast for return).
    kaiser_beta : float, default 8.6
        Kaiser window beta for the internal low-pass filter. 8.6 ≈ ~60 dB stopband.

    Returns
    -------
    y : np.ndarray
        Resampled array with the same shape as `x` except length along `axis`
        is ~ round(len * target_sr / sr).

    Notes
    -----
    - Uses polyphase FIR (`scipy.signal.resample_poly`) which applies the required
      anti-alias/imaging low-pass filtering automatically.
    - Ratio is reduced by gcd(sr, target_sr) → improved efficiency and numerical stability.
    - All-NaN/Inf inputs are handled via `nan_to_num` before processing; output is finite.
    - If SciPy is unavailable, raises a clear RuntimeError instead of silently degrading.
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "resample_audio requires SciPy (scipy.signal.resample_poly). "
            f"SciPy import failed with: {repr(_IMPORT_ERR)}"
        )

    if not isinstance(sr, int) or not isinstance(target_sr, int) or sr <= 0 or target_sr <= 0:
        raise ValueError(f"sr and target_sr must be positive integers, got sr={sr}, target_sr={target_sr}")

    # Fast path: only cast/clean when rates match
    if sr == target_sr:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)

    # Move the resample axis to the end for predictable behavior
    x_moved = np.moveaxis(x, axis, -1)

    # Clean numerics up front
    x_proc = np.nan_to_num(x_moved, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    # Reduce resampling ratio to simplest terms
    g = int(np.gcd(sr, target_sr))
    up, down = target_sr // g, sr // g

    # Polyphase FIR resample along the last axis
    # Use a Kaiser window; beta≈8.6 gives ~60 dB stopband attenuation (good general default).
    y = _resample_poly(x_proc, up=up, down=down, axis=-1, window=("kaiser", float(kaiser_beta)))

    # Return axis to original position and cast
    y = np.moveaxis(y, -1, axis)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)
    return y
