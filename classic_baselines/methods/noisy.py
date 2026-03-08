from __future__ import annotations

import numpy as np

from classic_baselines.config import DSPConfig

NAME = "noisy"
REQUIRES_PCA = False

def enhance(noisy_wav: np.ndarray, cfg: DSPConfig, *, pca_model: dict | None = None) -> np.ndarray:
    del cfg, pca_model
    return np.asarray(noisy_wav, dtype=np.float32)
