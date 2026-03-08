from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.methods.common import analyze_signal, apply_magnitude
from classic_baselines.pca import reconstruct_logmag_frames

NAME = "pca_subspace"
REQUIRES_PCA = True

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    if pca_model is None:
        raise ValueError("pca_subspace requires a PCA model.")
    state = analyze_signal(noisy_wav, cfg)
    logmag = np.log(np.maximum(state["mag"], cfg.eps)).T
    recon = reconstruct_logmag_frames(logmag, pca_model)
    magnitude = np.exp(recon).T
    return apply_magnitude(state["spec"], magnitude, cfg, state["length"])
