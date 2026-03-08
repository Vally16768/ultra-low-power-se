from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_gain

NAME = "spectral_gating"
REQUIRES_PCA = False

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    del pca_model
    state = analyze_signal(noisy_wav, cfg)
    noise_db = 10.0 * np.log10(np.maximum(state["noise_psd"], cfg.eps))[:, None]
    power_db = 10.0 * np.log10(np.maximum(state["power"], cfg.eps))
    soft = (power_db - noise_db - cfg.gating_threshold_db) / max(cfg.gating_slope_db, 1e-6)
    soft = np.clip(soft, 0.0, 1.0)
    if cfg.gating_freq_smooth > 1 or cfg.gating_time_smooth > 1:
        soft = uniform_filter(soft, size=(cfg.gating_freq_smooth, cfg.gating_time_smooth), mode="nearest")
    gain = cfg.gain_floor + (1.0 - cfg.gain_floor) * np.clip(soft, 0.0, 1.0)
    return apply_gain(state["spec"], gain, cfg, state["length"])
