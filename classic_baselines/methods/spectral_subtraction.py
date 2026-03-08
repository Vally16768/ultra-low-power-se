from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_gain

NAME = "spectral_subtraction"
REQUIRES_PCA = False

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    del pca_model
    state = analyze_signal(noisy_wav, cfg)
    floor = cfg.spectral_sub_floor * state["noise_psd"][:, None]
    enhanced_power = np.maximum(
        state["power"] - cfg.spectral_sub_alpha * state["noise_psd"][:, None],
        floor,
    )
    gain = np.sqrt(enhanced_power / np.maximum(state["power"], cfg.eps))
    return apply_gain(state["spec"], gain, cfg, state["length"])
