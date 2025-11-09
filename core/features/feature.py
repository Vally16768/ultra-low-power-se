#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature.py — Causal-friendly Track-1 front-end (16 kHz), STRICT (no fallbacks)
- Requires scipy.fft.dct if mel_ceps_keep > 0.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import soundfile as sf
import librosa


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 16000
    win_length: int = 320
    hop_length: int = 160
    n_fft: int = 320
    n_mels: int = 48
    fmin: float = 50.0
    fmax: Optional[float] = None
    mel_ceps_keep: int = 0
    yin_fmin: float = 50.0
    yin_fmax: float = 500.0
    # This factor is applied but we still enforce a strict >= 2.05 periods rule
    yin_frame_length_factor: float = 2.0
    pre_emph: float = 0.0
    peak_target: float = 0.95
    eps: float = 1e-8
    return_mag: bool = False


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
    if isinstance(path_or_wav, (str, Path)):
        wav, sr = sf.read(str(path_or_wav), dtype="float32", always_2d=False)
    else:
        wav = path_or_wav
        sr = target_sr

    if isinstance(wav, np.ndarray) and wav.ndim == 2:
        wav = wav.mean(axis=1)

    sr_in = sr if isinstance(path_or_wav, (str, Path)) else target_sr
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


def _dct_type_2_strict(mat: np.ndarray, keep: int) -> np.ndarray:
    # STRICT: require SciPy; no librosa fallback
    try:
        from scipy.fft import dct  # type: ignore
    except Exception as e:
        raise ImportError("scipy is required for DCT (mel_ceps_keep > 0) but is not installed.") from e
    return dct(mat, type=2, axis=-1, norm="ortho")[..., :keep].astype(np.float32)


def _yin_frame_length(sr: int, hop_length: int, yin_fmin: float, base_win_length: int, factor: float) -> int:
    """
    Choose a YIN frame_length that is:
      - at least base_win_length,
      - at least ceil(2.05 * sr / yin_fmin) (strictly > 2 periods),
      - odd (YIN prefers odd frame lengths),
      - and roughly respects the provided factor.
    """
    # Candidate from factor
    cand_fac = int(np.ceil(base_win_length * factor))
    # Strict >= 2.05 periods of the lowest pitch
    min_two_periods = int(np.ceil(2.05 * (sr / max(yin_fmin, 1e-6))))
    fl = max(base_win_length, cand_fac, min_two_periods)
    if fl % 2 == 0:
        fl += 1  # make odd
    return fl


class FeatureExtractor:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def from_file(
        self,
        path: str | Path,
        norm_stats: Optional[Dict[str, np.ndarray | float]] = None
    ) -> Dict[str, Any]:
        wav, _ = _ensure_mono_16k(path, self.cfg.sr)
        return self.from_wav(wav, norm_stats=norm_stats, source_path=str(path))

    def from_wav(
        self,
        wav: np.ndarray,
        norm_stats: Optional[Dict[str, np.ndarray | float]] = None,
        source_path: Optional[str] = None
    ) -> Dict[str, Any]:
        cfg = self.cfg

        if not isinstance(wav, np.ndarray):
            raise TypeError("from_wav expects a numpy array")
        wav, _ = _ensure_mono_16k(wav, cfg.sr)
        if wav.ndim != 1:
            raise ValueError("from_wav expects mono waveform")
        if wav.dtype != np.float32:
            raise ValueError("wav must be float32 mono @ 16 kHz")

        # Front-end: single, consistent pre-emphasis + peak normalization
        wav = _pre_emphasize(wav, cfg.pre_emph)
        wav = _peak_normalize(wav, cfg.peak_target, cfg.eps)

        # STFT (causal-friendly: center=False)
        n_fft_eff = max(int(cfg.n_fft), int(cfg.win_length))
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

        # Mel filterbank → log-mel (time axis last, then transpose to [T, M])
        fmax = cfg.fmax if cfg.fmax is not None else cfg.sr / 2
        mel_fb = librosa.filters.mel(
            sr=cfg.sr,
            n_fft=n_fft_eff,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=fmax,
            htk=True,
        )  # [M, F]
        mel_pow = mel_fb @ (mag ** 2)       # [M, T]
        mel_log = _safe_log(mel_pow, cfg.eps).T  # [T, M]
        T_mel = int(mel_log.shape[0])

        # Optional cepstra (strict SciPy DCT)
        mel_ceps = None
        if cfg.mel_ceps_keep and cfg.mel_ceps_keep > 0:
            mel_ceps = _dct_type_2_strict(mel_log, cfg.mel_ceps_keep)  # [T, K]

        # Pitch/voicing with STRICT YIN frame length
        yin_fl = _yin_frame_length(
            sr=cfg.sr,
            hop_length=cfg.hop_length,
            yin_fmin=cfg.yin_fmin,
            base_win_length=cfg.win_length,
            factor=cfg.yin_frame_length_factor,
        )
        f0 = librosa.yin(
            wav,
            fmin=cfg.yin_fmin,
            fmax=cfg.yin_fmax,
            sr=cfg.sr,
            frame_length=yin_fl,
            hop_length=cfg.hop_length,
        ).astype(np.float32)  # [T_f0]

        # --- Align lengths (YIN vs Mel frames may differ by >1); pad/trim to T_mel ---
        T_f0 = int(f0.shape[0])
        if T_f0 < T_mel:
            pad = T_mel - T_f0
            # pad with edge to keep continuity at the tail
            f0 = np.pad(f0, (0, pad), mode="edge")
        elif T_f0 > T_mel:
            f0 = f0[:T_mel]
        # Recompute vprob AFTER alignment to guarantee same length
        vprob = (f0 > 0.0).astype(np.float32)  # [T_mel]

        # Final feature assembly: [mel_log, f0, vprob, (opt) ceps]
        if mel_ceps is None:
            feats = np.concatenate([mel_log, f0[:, None], vprob[:, None]], axis=-1)  # [T, M+2]
        else:
            feats = np.concatenate([mel_log, f0[:, None], vprob[:, None], mel_ceps], axis=-1)  # [T, M+2+K]

        # Optional normalized view using provided stats (strict: stats must contain all keys)
        feats_norm = None
        used_stats = None
        if norm_stats is not None:
            mel_mean = np.asarray(norm_stats["mel_mean"], dtype=np.float32)   # [M]
            mel_std  = np.maximum(np.asarray(norm_stats["mel_std"], dtype=np.float32), 1e-6)  # [M]
            f0_mean  = float(norm_stats["f0_mean"])
            f0_std   = float(norm_stats["f0_std"]) if float(norm_stats["f0_std"]) > 0 else 1e-6

            mel_norm = (mel_log - mel_mean[None, :]) / mel_std[None, :]       # [T, M]
            f0n = np.zeros_like(f0, dtype=np.float32)
            vmask = f0 > 0.0
            if np.any(vmask):
                f0n[vmask] = (f0[vmask] - f0_mean) / f0_std
            if mel_ceps is None:
                feats_norm = np.concatenate([mel_norm, f0n[:, None], vprob[:, None]], axis=-1)
            else:
                feats_norm = np.concatenate([mel_norm, f0n[:, None], vprob[:, None], mel_ceps], axis=-1)

            used_stats = {
                "mel_mean": mel_mean,
                "mel_std": mel_std,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
            }

        meta = {"source": source_path or "<array>", "sr": cfg.sr, "n_fft_eff": n_fft_eff, "yin_frame_length": yin_fl}
        return {
            "feats": feats,                 # [T, D]
            "feats_norm": feats_norm,       # [T, D] or None
            "mel_log": mel_log,             # [T, M]
            "f0_hz": f0,                    # [T]
            "vprob": vprob,                 # [T]
            "mel_ceps": mel_ceps,           # [T, K] or None
            "mag": mag if cfg.return_mag else None,  # [F, T] or None
            "stats_used": used_stats,       # dict or None
            "meta": meta,
        }

    @staticmethod
    def save_norm_stats(
        path: str | Path,
        mel_mean: np.ndarray, mel_std: np.ndarray,
        f0_mean: float, f0_std: float
    ) -> None:
        np.savez(
            Path(path),
            mel_mean=np.asarray(mel_mean, dtype=np.float32),
            mel_std=np.asarray(mel_std, dtype=np.float32),
            f0_mean=np.float32(f0_mean),
            f0_std=np.float32(f0_std),
        )

    @staticmethod
    def load_norm_stats(path: str | Path) -> Dict[str, np.ndarray | float]:
        d = np.load(Path(path))
        return {
            "mel_mean": d["mel_mean"].astype(np.float32),
            "mel_std": d["mel_std"].astype(np.float32),
            "f0_mean": float(d["f0_mean"]),
            "f0_std": float(d["f0_std"]),
        }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path", type=str)
    ap.add_argument("--return-mag", action="store_true")
    ap.add_argument("--mel-ceps", type=int, default=0)
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
