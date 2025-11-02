from __future__ import annotations
import numpy as np

from .metrics_utils import to_mono, align, finite_or_default, clamp, LOG

def sisdr(s: np.ndarray, s_hat: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-invariant SDR in dB. Raises on error.
    """
    if not isinstance(s, np.ndarray) or not isinstance(s_hat, np.ndarray):
        raise TypeError("s and s_hat must be numpy arrays")
    s = s.astype(np.float32).reshape(-1)
    s_hat = s_hat.astype(np.float32).reshape(-1)
    n = min(len(s), len(s_hat))
    if n == 0:
        raise ValueError("Empty input")
    s = s[:n] - np.mean(s[:n])
    s_hat = s_hat[:n] - np.mean(s_hat[:n])
    denom = np.dot(s, s) + eps
    alpha = (np.dot(s_hat, s) + eps) / denom
    s_target = alpha * s
    e_noise = s_hat - s_target
    num = float(np.sum(s_target ** 2) + eps)
    den = float(np.sum(e_noise ** 2) + eps)
    return 10.0 * np.log10(num / den)

def sisdr_safe(s, s_hat, default: float = -30.0) -> float:
    """
    Training-safe SI-SDR:
      * folds to mono, aligns, handles all-silence (returns 0 dB)
      * clamps to [-60, 60]
      * never returns NaN/Inf; returns default on failure
    """
    try:
        s = to_mono(s)
        s_hat = to_mono(s_hat)
        s, s_hat = align(s, s_hat)
        if len(s) == 0:
            raise ValueError("Empty input")
        if np.allclose(s, 0.0) and np.allclose(s_hat, 0.0):
            return 0.0
        val = sisdr(s, s_hat)
        return clamp(finite_or_default(val, default, "SI-SDR"), -60.0, 60.0)
    except Exception as e:
        LOG.warning("SI-SDR failed: %s. Using default=%s.", e, default)
        return float(default)
