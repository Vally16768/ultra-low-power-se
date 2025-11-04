#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature.py — Causal-friendly Track-1 front-end (16 kHz)

Per 10 ms frame (causal; <= 40 ms algo delay):
  - 48 log-Mel bands (0..8 kHz), STFT with center=False (no lookahead)
  - Pitch F0 (Hz) via YIN + a binary voiced mask (periodicity proxy)
  - Optional DCT (cepstra) over log-Mel

API (training & inference):
  from core.features.feature import FeatureConfig, FeatureExtractor
  fx  = FeatureExtractor(FeatureConfig())
  out = fx.from_file("/path/to.wav", norm_stats=optional_stats_dict)

All arrays are float32. Audio is ALWAYS converted to mono @ 16 kHz.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import soundfile as sf
import librosa


# ------------------------------ Configuration -------------------------------- #

@dataclass(frozen=True)
class FeatureConfig:
    # I/O
    sr: int = 16000
    # Framing (causal-friendly)
    win_length: int = 320      # 20 ms @ 16 kHz
    hop_length: int = 160      # 10 ms @ 16 kHz
    n_fft: int = 320           # MUST be >= win_length; auto-upgraded at runtime if smaller
    # Mel
    n_mels: int = 48
    fmin: float = 50.0
    fmax: Optional[float] = None  # defaults to sr/2 if None
    # Optional cepstral DCT on log-Mel (set >0 to enable and keep K coeffs)
    mel_ceps_keep: int = 0     # 0 => disabled; e.g., 24 to enable
    # Pitch
    yin_fmin: float = 50.0
    yin_fmax: float = 500.0
    yin_frame_length_factor: float = 2.0  # baseline; we override adaptively to >2 periods
    # Pre-emphasis (off by default; can help SNR on some mics)
    pre_emph: float = 0.0
    # Peak normalization to prevent clipping oddities
    peak_target: float = 0.95
    # Numerical
    eps: float = 1e-8
    # Whether to also return STFT magnitude (off to save compute)
    return_mag: bool = False


# ------------------------------- Utilities ----------------------------------- #

def _pre_emphasize(x: np.ndarray, pre_emph: float) -> np.ndarray:
    if pre_emph <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - pre_emph * x[:-1]
    return y


def _safe_log(x: np.ndarray, eps: float) -> np.ndarray:
    return np.log(np.maximum(x, eps)).astype(np.float32)


def _ensure_mono_16k(path_or_wav: np.ndarray | str | Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load (or accept) audio, convert to mono, RESAMPLE to target_sr (ALWAYS),
    and return float32 in [-1, 1].
    """
    if isinstance(path_or_wav, (str, Path)):
        wav, sr = sf.read(str(path_or_wav), dtype="float32", always_2d=False)
    else:
        wav = path_or_wav
        sr = target_sr  # will be corrected below if needed

    # To mono
    if isinstance(wav, np.ndarray) and wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Determine input SR
    if not isinstance(path_or_wav, (str, Path)):
        sr_in = target_sr
    else:
        sr_in = sr

    # ALWAYS resample to target_sr
    wav = wav.astype(np.float32, copy=False)
    if sr_in != target_sr:
        wav = librosa.resample(wav, orig_sr=sr_in, target_sr=target_sr, res_type="kaiser_fast")
        sr_in = target_sr

    return wav, sr_in


def _peak_normalize(x: np.ndarray, peak_target: float, eps: float) -> np.ndarray:
    p = float(np.max(np.abs(x)) + eps)
    if p > 0.0:
        x = (peak_target / p) * x
    return x.astype(np.float32, copy=False)


def _dct_type_2(mat: np.ndarray, keep: int) -> np.ndarray:
    """Compute DCT-II over the last dimension and keep first K coeffs."""
    try:
        from scipy.fft import dct  # type: ignore
        return dct(mat, type=2, axis=-1, norm="ortho")[..., :keep].astype(np.float32)
    except Exception:
        # Fallback using librosa
        return librosa.core.dct(mat, type=2, axis=-1, norm="ortho")[..., :keep].astype(np.float32)


# ----------------------------- Core Extraction -------------------------------- #

class FeatureExtractor:
    """Causal-friendly feature front-end for Track 1."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    # ---- Public API ---------------------------------------------------------- #

    def from_file(self, path: str | Path,
                  norm_stats: Optional[Dict[str, np.ndarray | float]] = None
                 ) -> Dict[str, Any]:
        """Load audio from file and return feature pack (ALWAYS mono @ 16 kHz)."""
        wav, _ = _ensure_mono_16k(path, self.cfg.sr)
        return self.from_wav(wav, norm_stats=norm_stats, source_path=str(path))

    def from_wav(self, wav: np.ndarray,
                 norm_stats: Optional[Dict[str, np.ndarray | float]] = None,
                 source_path: Optional[str] = None
                ) -> Dict[str, Any]:
        """Extract features from a waveform array (will be forced to 16 kHz mono)."""
        cfg = self.cfg

        # Enforce mono @ 16 kHz no matter what came in
        if not isinstance(wav, np.ndarray):
            raise TypeError("from_wav expects a numpy array")
        wav, _ = _ensure_mono_16k(wav, cfg.sr)
        assert wav.ndim == 1, "from_wav expects mono waveform"
        assert wav.dtype == np.float32, "wav must be float32 mono @ 16 kHz"

        # Preproc
        wav = _pre_emphasize(wav, cfg.pre_emph)
        wav = _peak_normalize(wav, cfg.peak_target, cfg.eps)

        # Ensure n_fft >= win_length (auto-upgrade if needed)
        n_fft_eff = max(int(cfg.n_fft), int(cfg.win_length))

        # STFT mag (center=False keeps it causal-friendly)
        S = librosa.stft(
            wav,
            n_fft=n_fft_eff,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window="hann",
            center=False,
            pad_mode="constant",
        )
        mag = np.abs(S).astype(np.float32)  # [F, T]

        # Mel power -> log-Mel
        fmax = cfg.fmax if cfg.fmax is not None else cfg.sr / 2
        mel_fb = librosa.filters.mel(
            sr=cfg.sr, n_fft=n_fft_eff, n_mels=cfg.n_mels,
            fmin=cfg.fmin, fmax=fmax, htk=True
        )  # [n_mels, F]
        mel_pow = mel_fb @ (mag ** 2)                  # [n_mels, T]
        mel_log = _safe_log(mel_pow, cfg.eps).T        # [T, n_mels]

        # Optional DCT (cepstra) over Mel dimension
        mel_ceps = None
        if cfg.mel_ceps_keep and cfg.mel_ceps_keep > 0:
            mel_ceps = _dct_type_2(mel_log, cfg.mel_ceps_keep)  # [T, K]

        # Pitch + voicing (YIN); make frame_length safely > 2 periods at yin_fmin
        two_periods = 2.0 * cfg.sr / max(cfg.yin_fmin, 1e-3)  # e.g., 640 at 50 Hz
        yin_frame_length = int(max(cfg.win_length, np.ceil(two_periods) + 8))  # safety margin
        f0 = librosa.yin(
            wav, fmin=cfg.yin_fmin, fmax=cfg.yin_fmax, sr=cfg.sr,
            frame_length=yin_frame_length, hop_length=cfg.hop_length
        ).astype(np.float32)  # [T_y]
        vmask = ~np.isnan(f0)
        vprob = vmask.astype(np.float32)               # binary voiced prob proxy
        f0 = np.where(vmask, f0, 0.0).astype(np.float32)

        # Align lengths (YIN vs Mel frames may differ by 1)
        T = mel_log.shape[0]
        if len(f0) < T:
            pad = T - len(f0)
            f0 = np.pad(f0, (0, pad))
            vprob = np.pad(vprob, (0, pad))
        elif len(f0) > T:
            f0 = f0[:T]
            vprob = vprob[:T]

        # Assemble stacked feature matrix [T, D]
        # Convention: [logmel (48)] + [f0_hz (1)] + [vprob (1)] + [mel_ceps (K, optional)]
        parts = [mel_log, f0[:, None], vprob[:, None]]
        if mel_ceps is not None:
            parts.append(mel_ceps)
        feats = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # [T, D]

        # Optional normalization (keep vprob as-is)
        feats_norm = None
        used_stats: Dict[str, Any] = {}
        if norm_stats is not None:
            mel_mean = np.asarray(norm_stats.get("mel_mean", 0.0), dtype=np.float32)
            mel_std  = np.asarray(norm_stats.get("mel_std", 1.0), dtype=np.float32)
            f0_mean  = np.float32(norm_stats.get("f0_mean", 0.0))
            f0_std   = np.float32(norm_stats.get("f0_std", 1.0))
            # Apply to mel_log and f0; skip vprob and (optionally) mel_ceps
            mel_norm = (mel_log - mel_mean[None, :]) / np.maximum(mel_std[None, :], self.cfg.eps)
            f0n = (f0 - f0_mean) / max(float(f0_std), float(self.cfg.eps))
            parts_n = [mel_norm, f0n[:, None], vprob[:, None]]
            if mel_ceps is not None:
                parts_n.append(mel_ceps)  # usually left unnormalized
            feats_norm = np.concatenate(parts_n, axis=-1).astype(np.float32, copy=False)
            used_stats = {
                "mel_mean": mel_mean, "mel_std": mel_std,
                "f0_mean": float(f0_mean), "f0_std": float(f0_std),
            }

        meta = {
            "sr": cfg.sr,
            "frame_hop": cfg.hop_length,
            "frame_len": cfg.win_length,
            "n_fft_req": cfg.n_fft,
            "n_fft_eff": n_fft_eff,         # actual FFT size used
            "n_mels": cfg.n_mels,
            "mel_ceps_keep": cfg.mel_ceps_keep,
            "source_path": source_path,
            "T": T,
            "D": feats.shape[1],
            "config": asdict(cfg),
        }

        pack: Dict[str, Any] = {
            "feats": feats,                # [T, D] float32 (unnormalized)
            "feats_norm": feats_norm,      # [T, D] float32 or None
            "mel_log": mel_log,            # [T, n_mels]
            "f0_hz": f0,                   # [T]
            "vprob": vprob,                # [T]
            "mel_ceps": mel_ceps,          # [T, K] or None
            "mag": mag if cfg.return_mag else None,  # [F, T] or None
            "stats_used": used_stats,      # dict (if normalization applied)
            "meta": meta,
        }
        return pack

    # ---- Helpers for stats persistence -------------------------------------- #

    @staticmethod
    def save_norm_stats(path: str | Path,
                        mel_mean: np.ndarray, mel_std: np.ndarray,
                        f0_mean: float, f0_std: float) -> None:
        np.savez(Path(path),
                 mel_mean=np.asarray(mel_mean, dtype=np.float32),
                 mel_std=np.asarray(mel_std, dtype=np.float32),
                 f0_mean=np.float32(f0_mean),
                 f0_std=np.float32(f0_std))

    @staticmethod
    def load_norm_stats(path: str | Path) -> Dict[str, np.ndarray | float]:
        d = np.load(Path(path))
        return {
            "mel_mean": d["mel_mean"].astype(np.float32),
            "mel_std": d["mel_std"].astype(np.float32),
            "f0_mean": float(d["f0_mean"]),
            "f0_std": float(d["f0_std"]),
        }


# ------------------------------ Quick CLI test -------------------------------- #

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path", type=str, help="Path to a .wav file (any SR; will be resampled to 16 kHz)")
    ap.add_argument("--return-mag", action="store_true", help="Also return STFT magnitude (debug)")
    ap.add_argument("--mel-ceps", type=int, default=0, help="If >0, keep that many DCT coeffs of log-Mel")
    args = ap.parse_args()

    cfg = FeatureConfig(return_mag=args.return_mag, mel_ceps_keep=args.mel_ceps)
    fx = FeatureExtractor(cfg)
    pack = fx.from_file(args.wav_path)

    summary = {
        "feats.shape": list(pack["feats"].shape),
        "mel_log.shape": list(pack["mel_log"].shape),
        "f0_hz.shape": list(pack["f0_hz"].shape),
        "vprob.shape": list(pack["vprob"].shape),
        "mel_ceps.shape": (None if pack["mel_ceps"] is None else list(pack["mel_ceps"].shape)),
        "mag.shape": (None if pack["mag"] is None else [int(x) for x in pack["mag"].shape]),
        "meta": pack["meta"],
    }
    print(json.dumps(summary, indent=2))
