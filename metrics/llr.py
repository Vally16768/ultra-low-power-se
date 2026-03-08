from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from metrics.metrics_utils import align, extract_overlapped_windows, hann_window, to_mono

_EPS = np.finfo(np.float64).eps

def _lpcoeff(speech_frame: np.ndarray, model_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return LPC coefficients and autocorrelation using Levinson-Durbin recursion."""
    speech_frame = np.asarray(speech_frame, dtype=np.float64)
    r = np.zeros(model_order + 1, dtype=np.float64)
    for k in range(model_order + 1):
        if k == 0:
            r[k] = np.sum(speech_frame * speech_frame)
        else:
            r[k] = np.sum(speech_frame[:-k] * speech_frame[k:])

    a = np.ones(model_order, dtype=np.float64)
    a_prev = np.ones(model_order, dtype=np.float64)
    reflection = np.zeros(model_order, dtype=np.float64)
    error = np.zeros(model_order + 1, dtype=np.float64)
    error[0] = max(r[0], _EPS)

    for i in range(model_order):
        a_prev[:i] = a[:i]
        sum_term = np.sum(a_prev[:i] * r[i:0:-1]) if i > 0 else 0.0
        reflection[i] = (r[i + 1] - sum_term) / max(error[i], _EPS)
        a[i] = reflection[i]
        if i > 0:
            a[:i] = a_prev[:i] - reflection[i] * a_prev[i - 1 :: -1]
        error[i + 1] = max((1.0 - reflection[i] * reflection[i]) * error[i], _EPS)

    lpc = np.ones(model_order + 1, dtype=np.float64)
    lpc[1:] = -a
    return lpc, r

def log_likelihood_ratio(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    used_for_composite: bool = False,
    frame_len: float = 0.03,
    overlap: float = 0.75,
) -> float:
    """Compute the trimmed-mean LLR distortion."""
    clean, processed = align(to_mono(clean_speech), to_mono(processed_speech))
    winlength = max(1, int(round(frame_len * fs)))
    skiprate = max(1, int(np.floor((1.0 - overlap) * frame_len * fs)))
    if clean.size < winlength:
        pad = winlength - clean.size
        clean = np.pad(clean, (0, pad))
        processed = np.pad(processed, (0, pad))

    order = 10 if fs < 10000 else 16
    window = hann_window(winlength)
    clean_frames = extract_overlapped_windows(clean + _EPS, winlength, winlength - skiprate, window)
    processed_frames = extract_overlapped_windows(processed + _EPS, winlength, winlength - skiprate, window)
    n_frames = clean_frames.shape[0]
    distortions = []
    for idx in range(max(1, n_frames - 1)):
        a_clean, r_clean = _lpcoeff(clean_frames[idx], order)
        a_proc, _ = _lpcoeff(processed_frames[idx], order)
        numer = float(a_proc @ toeplitz(r_clean).dot(a_proc.T))
        denom = float(a_clean @ toeplitz(r_clean).dot(a_clean.T))
        frac = numer / max(denom, _EPS)
        if not np.isfinite(frac) or frac <= 0.0:
            frac = np.exp(2.0)
        dist = float(np.log(frac))
        if not used_for_composite and dist > 2.0:
            dist = 2.0
        if not np.isfinite(dist):
            dist = 2.0
        distortions.append(dist)

    distortion = np.sort(np.asarray(distortions, dtype=np.float64))
    keep = max(1, int(round(len(distortion) * 0.95)))
    return float(np.mean(distortion[:keep]))
