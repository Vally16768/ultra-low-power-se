from __future__ import annotations

import numpy as np
from scipy.special import ive

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_gain, decision_directed_prior

NAME = "mmse_stsa"
REQUIRES_PCA = False

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    del pca_model
    state = analyze_signal(noisy_wav, cfg)
    gamma = np.maximum(state["power"] / state["noise_psd"][:, None], 1e-12)
    gain = np.zeros_like(gamma, dtype=np.float32)
    prev_gain = np.ones(gamma.shape[0], dtype=np.float32)
    prev_gamma = np.maximum(gamma[:, 0], 1.0)
    for idx in range(gamma.shape[1]):
        xi = decision_directed_prior(gamma[:, idx], prev_gain, prev_gamma, cfg.mmse_alpha)
        nu = np.maximum(gamma[:, idx] * xi / (1.0 + xi), 1e-12)
        half_nu = 0.5 * nu
        numer = np.sqrt(np.pi) * np.sqrt(nu)
        bessel = (1.0 + nu) * ive(0, half_nu) + nu * ive(1, half_nu)
        mmse_gain = 0.5 * numer / gamma[:, idx] * bessel
        gain[:, idx] = np.clip(mmse_gain, cfg.gain_floor, 1.0)
        prev_gain = gain[:, idx]
        prev_gamma = gamma[:, idx]
    return apply_gain(state["spec"], gain, cfg, state["length"])
