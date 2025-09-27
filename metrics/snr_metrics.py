import numpy as np

_EPS = 1e-12

def _snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """
    SNR(clean, test) = 10*log10( sum(clean^2) / sum((clean - test)^2) )
    Așteaptă semnale aliniate și aceeași rată de eșantionare.
    """
    c = clean.astype(np.float64).flatten()
    t = test.astype(np.float64).flatten()
    nrg = np.sum(c * c) + _EPS
    err = c - t
    den = np.sum(err * err) + _EPS
    return 10.0 * np.log10(nrg / den)

def delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """
    ΔSNR = SNR(clean, enhanced) - SNR(clean, noisy)
    Valori pozitive înseamnă îmbunătățire.
    """
    return _snr_db(clean, enhanced) - _snr_db(clean, noisy)

def snr_noisy(clean: np.ndarray, noisy: np.ndarray) -> float:
    return _snr_db(clean, noisy)

def snr_enhanced(clean: np.ndarray, enhanced: np.ndarray) -> float:
    return _snr_db(clean, enhanced)
