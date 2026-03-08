from __future__ import annotations

from typing import Dict

import numpy as np

from metrics.llr import log_likelihood_ratio
from metrics.pesq import pesq_score
from metrics.segsnr import segmental_snr
from metrics.sisdr import sisdr
from metrics.snr import snr_enhanced, snr_noisy
from metrics.stoi import stoi_score
from metrics.wss import weighted_spectral_slope

_EPS = 1e-8

def _align_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    n = min(len(a), len(b))
    if n <= 0:
        raise ValueError("Cannot align empty signals.")
    return a[:n], b[:n]

def _clip_mos(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))

def si_snr(ref: np.ndarray, est: np.ndarray, eps: float = _EPS) -> float:
    ref, est = _align_pair(ref, est)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))

def composite_scores(clean: np.ndarray, processed: np.ndarray, sr: int) -> Dict[str, float]:
    clean_proc, processed = _align_pair(clean, processed)
    wss = float(weighted_spectral_slope(clean_proc, processed, sr))
    llr = float(log_likelihood_ratio(clean_proc, processed, sr, used_for_composite=True))
    seg_snr = float(segmental_snr(clean_proc, processed, sr))
    pesq = float(pesq_score(clean_proc, processed, sr))

    csig = _clip_mos(3.093 - 1.029 * llr + 0.603 * pesq - 0.009 * wss)
    cbak = _clip_mos(1.634 + 0.478 * pesq - 0.007 * wss + 0.063 * seg_snr)
    covl = _clip_mos(1.594 + 0.805 * pesq - 0.512 * llr - 0.007 * wss)
    return {
        "PESQ": pesq,
        "WSS": wss,
        "LLR": llr,
        "SEG_SNR": seg_snr,
        "CSIG": csig,
        "CBAK": cbak,
        "COVL": covl,
    }

def evaluate_pair_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    enhanced_path: str | None = None,
) -> Dict[str, float]:
    """Return the canonical intrusive metric set for a clean/noisy/enhanced triplet."""
    clean_noisy, noisy = _align_pair(clean, noisy)
    clean_enh, enhanced = _align_pair(clean, enhanced)

    snr_in = float(snr_noisy(clean_noisy, noisy))
    snr_out = float(snr_enhanced(clean_enh, enhanced))
    si_snr_in = float(si_snr(clean_noisy, noisy))
    si_snr_out = float(si_snr(clean_enh, enhanced))
    metrics = {
        "SNR_IN": snr_in,
        "SNR_OUT": snr_out,
        "SNRi": snr_out - snr_in,
        "SI_SDR": float(sisdr(clean_enh, enhanced)),
        "SI-SNRi": si_snr_out - si_snr_in,
        "STOI": float(stoi_score(clean_enh, enhanced, sr, extended=False)),
    }
    metrics.update(composite_scores(clean_enh, enhanced, sr))

    if enhanced_path is not None:
        from metrics.dnsmos import dnsmos_wav

        dns = dnsmos_wav(str(enhanced_path))
        metrics["DNSMOS_SIG"] = float(dns["mos_sig"])
        metrics["DNSMOS_BAK"] = float(dns["mos_bak"])
        metrics["DNSMOS_OVR"] = float(dns["mos_ovr"])

    return metrics
