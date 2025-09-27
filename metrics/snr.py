import numpy as np
def snr(signal: np.ndarray, noise: np.ndarray) -> float:
    s = np.mean(signal**2) + 1e-12
    n = np.mean(noise**2) + 1e-12
    return 10.0 * np.log10(s/n)

def delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    # ΔSNR = SNR(clean, noisy-clean) -> SNR(clean, enhanced-clean)
    n_in  = noisy[:len(clean)] - clean[:len(noisy)]
    n_out = enhanced[:len(clean)] - clean[:len(enhanced)]
    return snr(clean, n_out) - snr(clean, n_in)
