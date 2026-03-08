from __future__ import annotations

from classic_baselines.methods import (
    logmmse,
    mmse_stsa,
    noisy,
    pca_subspace,
    pca_wiener,
    spectral_gating,
    spectral_subtraction,
    wiener,
)

_MODULES = (
    noisy,
    wiener,
    spectral_subtraction,
    spectral_gating,
    mmse_stsa,
    logmmse,
    pca_subspace,
    pca_wiener,
)

METHODS = {module.NAME: module for module in _MODULES}

def get_method(name: str):
    if name not in METHODS:
        raise KeyError(f"Unknown classic baseline method: {name}")
    return METHODS[name]

def list_methods() -> list[str]:
    return list(METHODS.keys())
