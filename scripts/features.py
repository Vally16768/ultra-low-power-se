from __future__ import annotations
import numpy as np
from scipy.signal import stft, istft

def stft_mag(x, fs=16000, n_fft=512, hop=128):
    assert x.ndim == 1, "stft_mag expects mono waveform"
    f, t, Z = stft(x, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    return Z, np.abs(Z)

def ideal_ratio_mask(m_noisy, m_clean, eps=1e-8):
    return np.clip((m_clean**2) / (m_noisy**2 + eps), 0.0, 1.0)

def inverse_from_mask(x_noisy, mask, fs=16000, n_fft=512, hop=128):
    # Recompute complex STFT for phase; apply mask on magnitude

    _, _, Z = stft(x_noisy, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    assert Z.shape == mask.shape, f"Mask shape {mask.shape} must equal STFT shape {Z.shape}"
    Z_est = Z * mask
    _, y = istft(Z_est, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return y.astype(np.float32)
