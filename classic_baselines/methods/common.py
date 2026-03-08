from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.noise import estimate_noise_psd
from classic_baselines.stft import istft_synthesis, stft_analysis

def analyze_signal(noisy_wav: np.ndarray, cfg: DSPConfig) -> dict:
    spec = stft_analysis(noisy_wav, cfg)
    mag = np.abs(spec)
    power = np.maximum(mag ** 2, cfg.eps)
    noise_psd = estimate_noise_psd(power, cfg.noise_frames)
    return {
        "spec": spec,
        "mag": mag,
        "power": power,
        "noise_psd": np.maximum(noise_psd, cfg.eps),
        "length": len(noisy_wav),
    }

def decision_directed_prior(
    gamma_t: np.ndarray,
    prev_gain: np.ndarray,
    prev_gamma: np.ndarray,
    alpha: float,
) -> np.ndarray:
    xi = alpha * (prev_gain ** 2) * prev_gamma + (1.0 - alpha) * np.maximum(gamma_t - 1.0, 0.0)
    return np.maximum(xi, 1e-12)

def apply_gain(noisy_spec: np.ndarray, gain: np.ndarray, cfg: DSPConfig, length: int) -> np.ndarray:
    gain = np.clip(np.asarray(gain, dtype=np.float32), cfg.gain_floor, 1.0)
    enhanced = noisy_spec * gain
    return istft_synthesis(enhanced, cfg, length)

def apply_magnitude(noisy_spec: np.ndarray, magnitude: np.ndarray, cfg: DSPConfig, length: int) -> np.ndarray:
    phase = np.exp(1j * np.angle(noisy_spec))
    enhanced = np.asarray(magnitude, dtype=np.float32) * phase
    return istft_synthesis(enhanced, cfg, length)
