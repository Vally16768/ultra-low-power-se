from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class DSPConfig:
    sr: int = 16000
    n_fft: int = 320
    win_length: int = 320
    hop_length: int = 160
    center: bool = False
    noise_frames: int = 20
    gain_floor: float = 0.05
    wiener_alpha: float = 0.98
    spectral_sub_alpha: float = 2.0
    spectral_sub_floor: float = 0.02
    gating_threshold_db: float = 6.0
    gating_slope_db: float = 12.0
    gating_time_smooth: int = 3
    gating_freq_smooth: int = 5
    mmse_alpha: float = 0.98
    pca_max_components: int = 64
    pca_var_threshold: float = 0.95
    eps: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)
