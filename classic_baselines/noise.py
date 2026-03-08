from __future__ import annotations

import numpy as np

def estimate_noise_psd(power_spectrogram: np.ndarray, num_frames: int = 20) -> np.ndarray:
    """Estimate stationary noise PSD from the first frames using the median."""
    if power_spectrogram.ndim != 2:
        raise ValueError("Expected power_spectrogram with shape [F, T].")
    frames = max(1, min(int(num_frames), power_spectrogram.shape[1]))
    return np.median(np.maximum(power_spectrogram[:, :frames], 1e-12), axis=1)
