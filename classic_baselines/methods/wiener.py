from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_gain, decision_directed_prior

NAME = "wiener"
REQUIRES_PCA = False

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    del pca_model
    state = analyze_signal(noisy_wav, cfg)
    gamma = state["power"] / state["noise_psd"][:, None]
    gain = np.zeros_like(gamma, dtype=np.float32)
    prev_gain = np.ones(gamma.shape[0], dtype=np.float32)
    prev_gamma = np.maximum(gamma[:, 0], 1.0)
    for idx in range(gamma.shape[1]):
        xi = decision_directed_prior(gamma[:, idx], prev_gain, prev_gamma, cfg.wiener_alpha)
        gain[:, idx] = np.clip(xi / (1.0 + xi), cfg.gain_floor, 1.0)
        prev_gain = gain[:, idx]
        prev_gamma = gamma[:, idx]
    return apply_gain(state["spec"], gain, cfg, state["length"])
