from __future__ import annotations

import librosa
import numpy as np

from classic_baselines.config import DSPConfig

def stft_analysis(wav: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    return librosa.stft(
        np.asarray(wav, dtype=np.float32),
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        win_length=int(cfg.win_length),
        window="hann",
        center=bool(cfg.center),
        pad_mode="constant",
    )

def istft_synthesis(spec: np.ndarray, cfg: DSPConfig, length: int) -> np.ndarray:
    wav = librosa.istft(
        spec,
        hop_length=int(cfg.hop_length),
        win_length=int(cfg.win_length),
        window="hann",
        center=bool(cfg.center),
        length=int(length),
    ).astype(np.float32)
    if wav.shape[0] < length:
        wav = np.pad(wav, (0, length - wav.shape[0]))
    return wav[:length]
