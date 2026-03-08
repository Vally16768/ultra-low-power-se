from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_gain
from classic_baselines.pca import reconstruct_logmag_frames

NAME = "pca_wiener"
REQUIRES_PCA = True

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    if pca_model is None:
        raise ValueError("pca_wiener requires a PCA model.")
    state = analyze_signal(noisy_wav, cfg)
    logmag = np.log(np.maximum(state["mag"], cfg.eps)).T
    recon = reconstruct_logmag_frames(logmag, pca_model)
    speech_power = np.exp(2.0 * recon).T
    gain = speech_power / (speech_power + state["noise_psd"][:, None] + cfg.eps)
    return apply_gain(state["spec"], gain, cfg, state["length"])
