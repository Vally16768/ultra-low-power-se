from __future__ import annotations
import numpy as np

from .metrics_utils import to_mono, align, finite_or_default, clamp, LOG

_EPS = 1e-12

def _snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """
    SNR(clean, test) = 10*log10( sum(clean^2) / sum((clean - test)^2) ).
    Raises on error; expects arrays aligned externally.
    """
    c = clean.astype(np.float64).ravel()
    t = test.astype(np.float64).ravel()
    n = min(len(c), len(t))
    if n == 0:
        raise ValueError("Empty input")
    c = c[:n]
    t = t[:n]
    nrg = float(np.sum(c * c) + _EPS)
    err = c - t
    den = float(np.sum(err * err) + _EPS)
    return 10.0 * np.log10(nrg / den)


def snr_noisy(clean: np.ndarray, noisy: np.ndarray) -> float:
    return _snr_db(clean, noisy)


def snr_enhanced(clean: np.ndarray, enhanced: np.ndarray) -> float:
    return _snr_db(clean, enhanced)


def delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """ΔSNR = SNR(clean, enhanced) - SNR(clean, noisy)."""
    n = min(len(clean), len(noisy), len(enhanced))
    if n == 0:
        raise ValueError("Empty input")
    c = clean[:n]
    n_ = noisy[:n]
    e_ = enhanced[:n]
    return _snr_db(c, e_) - _snr_db(c, n_)


def snr_noisy_safe(clean, noisy, default: float = 0.0) -> float:
    try:
        c = to_mono(clean); n = to_mono(noisy)
        c, n = align(c, n)
        if len(c) == 0:
            raise ValueError("Empty input")
        val = snr_noisy(c, n)
        return clamp(finite_or_default(val, default, "SNR_NOISY"), -60.0, 60.0)
    except Exception as e:
        LOG.warning("SNR (noisy) failed: %s. Using default=%s.", e, default)
        return float(default)


def snr_enhanced_safe(clean, enhanced, default: float = 0.0) -> float:
    try:
        c = to_mono(clean); e = to_mono(enhanced)
        c, e = align(c, e)
        if len(c) == 0:
            raise ValueError("Empty input")
        val = snr_enhanced(c, e)
        return clamp(finite_or_default(val, default, "SNR_ENH"), -60.0, 60.0)
    except Exception as e:
        LOG.warning("SNR (enhanced) failed: %s. Using default=%s.", e, default)
        return float(default)


def delta_snr_safe(clean, noisy, enhanced, default: float = 0.0) -> float:
    try:
        c = to_mono(clean); n = to_mono(noisy); e = to_mono(enhanced)
        m = min(len(c), len(n), len(e))
        if m == 0:
            raise ValueError("Empty input")
        c, n, e = c[:m], n[:m], e[:m]
        val = delta_snr(c, n, e)
        return clamp(finite_or_default(val, default, "DELTA_SNR"), -60.0, 60.0)
    except Exception as e:
        LOG.warning("ΔSNR failed: %s. Using default=%s.", e, default)
        return float(default)
