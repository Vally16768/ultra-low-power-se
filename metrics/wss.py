from __future__ import annotations

import numpy as np
from scipy.signal import stft

from metrics.metrics_utils import align, hann_window, to_mono

_EPS = np.finfo(np.float64).eps
_ALPHA = 0.95
_KMAX = 20.0
_KLOCMAX = 1.0
_CENT_FREQ = np.array([
    50.0000, 120.0000, 190.0000, 260.0000, 330.0000, 400.0000, 470.0000,
    540.0000, 617.3720, 703.3780, 798.7170, 904.1280, 1020.3800, 1148.3000,
    1288.7200, 1442.5400, 1610.7000, 1794.1600, 1993.9300, 2211.0800,
    2446.7100, 2701.9700, 2978.0400, 3276.1700, 3597.6300,
], dtype=np.float64)
_BANDWIDTH = np.array([
    70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 77.3724,
    86.0056, 95.3398, 105.4110, 116.2560, 127.9140, 140.4230, 153.8230,
    168.1540, 183.4570, 199.7760, 217.1530, 235.6310, 255.2550, 276.0720,
    298.1260, 321.4650, 346.1360,
], dtype=np.float64)

def _find_local_peaks(slope: np.ndarray, energy: np.ndarray) -> np.ndarray:
    peaks = np.zeros_like(slope)
    num_crit = len(energy)
    for idx in range(len(slope)):
        pos = idx
        if slope[idx] > 0:
            while pos < num_crit - 1 and slope[pos] > 0:
                pos += 1
            peaks[idx] = energy[pos - 1]
        else:
            while pos >= 0 and slope[pos] <= 0:
                pos -= 1
            peaks[idx] = energy[pos + 1]
    return peaks

def weighted_spectral_slope(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    frame_len: float = 0.03,
    overlap: float = 0.75,
) -> float:
    """Compute the trimmed-mean WSS distortion."""
    clean, processed = align(to_mono(clean_speech), to_mono(processed_speech))
    winlength = max(1, int(round(frame_len * fs)))
    skiprate = max(1, int(np.floor((1.0 - overlap) * frame_len * fs)))
    if clean.size < winlength:
        pad = winlength - clean.size
        clean = np.pad(clean, (0, pad))
        processed = np.pad(processed, (0, pad))

    max_freq = fs / 2.0
    n_fft = int(2 ** np.ceil(np.log2(2 * winlength)))
    n_fftby2 = n_fft // 2
    bw_min = _BANDWIDTH[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))
    j = np.arange(n_fftby2, dtype=np.float64)
    crit_filter = np.zeros((_CENT_FREQ.size, n_fftby2), dtype=np.float64)

    for idx, (cent_freq, bandwidth) in enumerate(zip(_CENT_FREQ, _BANDWIDTH)):
        f0 = (cent_freq / max_freq) * n_fftby2
        bw = (bandwidth / max_freq) * n_fftby2
        norm = np.log(bw_min) - np.log(bandwidth)
        crit = np.exp(-11.0 * (((j - np.floor(f0)) / bw) ** 2) + norm)
        crit_filter[idx] = crit * (crit > min_factor)

    num_frames = int(len(clean) / skiprate - (winlength / skiprate))
    if num_frames <= 0:
        num_frames = 1
    usable = int(num_frames) * skiprate + int(winlength - skiprate)
    clean = clean[:usable]
    processed = processed[:usable]
    window = hann_window(winlength)
    scale = np.sqrt(1.0 / np.sum(window) ** 2)

    _, _, clean_stft = stft(
        clean, fs=fs, window=window, nperseg=winlength, noverlap=winlength - skiprate,
        nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False
    )
    _, _, proc_stft = stft(
        processed, fs=fs, window=window, nperseg=winlength, noverlap=winlength - skiprate,
        nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False
    )

    clean_spec = (np.abs(clean_stft) / scale) ** 2
    proc_spec = (np.abs(proc_stft) / scale) ** 2
    clean_spec = clean_spec[:-1, :]
    proc_spec = proc_spec[:-1, :]

    clean_energy = crit_filter.dot(clean_spec)
    proc_energy = crit_filter.dot(proc_spec)
    log_clean = 10.0 * np.log10(np.maximum(clean_energy, _EPS))
    log_proc = 10.0 * np.log10(np.maximum(proc_energy, _EPS))
    log_clean = np.maximum(log_clean, -100.0)
    log_proc = np.maximum(log_proc, -100.0)

    clean_slope = np.diff(log_clean, axis=0)
    proc_slope = np.diff(log_proc, axis=0)
    clean_max = np.max(log_clean, axis=0)
    proc_max = np.max(log_proc, axis=0)

    clean_peaks = np.zeros_like(clean_slope)
    proc_peaks = np.zeros_like(proc_slope)
    for idx in range(clean_slope.shape[1]):
        clean_peaks[:, idx] = _find_local_peaks(clean_slope[:, idx], log_clean[:, idx])
        proc_peaks[:, idx] = _find_local_peaks(proc_slope[:, idx], log_proc[:, idx])

    wmax_clean = _KMAX / (_KMAX + clean_max - log_clean[:-1, :])
    wloc_clean = _KLOCMAX / (_KLOCMAX + clean_peaks - log_clean[:-1, :])
    w_clean = wmax_clean * wloc_clean
    wmax_proc = _KMAX / (_KMAX + proc_max - log_proc[:-1, :])
    wloc_proc = _KLOCMAX / (_KLOCMAX + proc_peaks - log_proc[:-1, :])
    w_proc = wmax_proc * wloc_proc
    weights = 0.5 * (w_clean + w_proc)

    distortion = np.sum(weights * (clean_slope - proc_slope) ** 2, axis=0) / np.sum(weights, axis=0)
    distortion = np.nan_to_num(distortion, nan=0.0, posinf=0.0, neginf=0.0)
    distortion = np.sort(distortion)
    keep = max(1, int(round(len(distortion) * _ALPHA)))
    return float(np.mean(distortion[:keep]))
