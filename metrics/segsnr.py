from __future__ import annotations

import numpy as np

from metrics.metrics_utils import align, extract_overlapped_windows, hann_window, to_mono

_EPS = np.finfo(np.float64).eps

def segmental_snr(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    frame_len: float = 0.03,
    overlap: float = 0.75,
) -> float:
    """Compute clipped segmental SNR using the standard Loizou-style setup."""
    clean, processed = align(to_mono(clean_speech), to_mono(processed_speech))
    winlength = max(1, int(round(frame_len * fs)))
    skiprate = max(1, int(np.floor((1.0 - overlap) * frame_len * fs)))
    if clean.size < winlength:
        pad = winlength - clean.size
        clean = np.pad(clean, (0, pad))
        processed = np.pad(processed, (0, pad))

    window = hann_window(winlength)
    clean_frames = extract_overlapped_windows(clean, winlength, winlength - skiprate, window)
    processed_frames = extract_overlapped_windows(processed, winlength, winlength - skiprate, window)

    signal_energy = np.sum(clean_frames ** 2, axis=-1)
    noise_energy = np.sum((clean_frames - processed_frames) ** 2, axis=-1)
    segmental = 10.0 * np.log10(signal_energy / (noise_energy + _EPS) + _EPS)
    segmental = np.clip(segmental, -10.0, 35.0)
    if segmental.size > 1:
        segmental = segmental[:-1]
    return float(np.mean(segmental))
