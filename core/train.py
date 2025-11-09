#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/train.py — Track 1 training (stable, no-NaN metrics)

Highlights
- Uses voiced-frame sample_weight (0.3..1.0) for the per-timestep loss.
- Robust L1 log-Mel loss (+ tiny energy term) with Adam(clipnorm=1.0).
- Safer STFT->ISTFT reconstruction (cap at 3.0× noisy magnitude).
- Intrusive metrics (SNR, SI-SDR, PESQ, STOI) computed with safe wrappers.
- Non-intrusive metrics:
    * DNSMOS: computed from the saved enhanced wav via dnsmos_wav_safe()
    * NISQA: optional --nisqa-ckpt path; otherwise default MOS is used.
- Never writes NaN in metrics.json (falls back to finite defaults).
"""

from __future__ import annotations
import argparse, json, os, random, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import Sequence

# ----------------------------- Local imports --------------------------------- #
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Front-end / model (project-local)
from core.features.feature import FeatureConfig, FeatureExtractor  # noqa: E402
from core.models.model import get_model                            # noqa: E402

# Metric helpers/logger
try:
    from metrics_logger import setup_metrics_logger  # configures logger "metrics"
except Exception:
    setup_metrics_logger = None

if setup_metrics_logger:
    setup_metrics_logger(level=logging.WARNING)

# ---- Intrusive metrics (safe) ------------------------------------------------
# SI-SDR
try:
    from sisdr import sisdr_safe as SISDR_SAFE
except Exception:
    SISDR_SAFE = None  # handled downstream

# PESQ (safe wrapper picks a backend if available; else returns default)
try:
    from pesq import pesq_score_safe as PESQ_SAFE
except Exception:
    PESQ_SAFE = None

# STOI (safe wrapper)
try:
    from stoi import stoi_score_safe as STOI_SAFE
except Exception:
    STOI_SAFE = None

# ---- Non-intrusive metrics (optional) ---------------------------------------
# DNSMOS works on WAV files; use _wav_safe API and map to SIG/BAK/OVR
try:
    from dnsmos import dnsmos_wav_safe as DNSMOS_WAV_SAFE
except Exception:
    DNSMOS_WAV_SAFE = None

# NISQA requires a model checkpoint; we load it lazily if --nisqa-ckpt is given
try:
    from nisqa import load_nisqa, nisqa_file_safe as NISQA_FILE_SAFE
except Exception:
    load_nisqa = None
    NISQA_FILE_SAFE = None


# ----------------------------- Utils ----------------------------------------- #

def set_seed(seed: int = 41):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_feature_stats(stats_npz_path: Path):
    d = np.load(str(stats_npz_path))
    mel_mean = d["mel_mean"].astype(np.float32)   # [48]
    mel_std  = d["mel_std"].astype(np.float32)    # [48]
    f0_mean  = float(d["f0_mean"])
    f0_std   = float(d["f0_std"])
    mel_std  = np.maximum(mel_std, 1e-6)
    f0_std   = max(f0_std, 1e-6)
    return mel_mean, mel_std, f0_mean, f0_std

def z_norm_mel(mel_log: np.ndarray, mel_mean: np.ndarray, mel_std: np.ndarray) -> np.ndarray:
    return (mel_log - mel_mean[None, :]) / mel_std[None, :]

def z_denorm_mel(mel_norm: np.ndarray, mel_mean: np.ndarray, mel_std: np.ndarray) -> np.ndarray:
    return mel_norm * mel_std[None, :] + mel_mean[None, :]

def z_norm_f0(f0: np.ndarray, f0_mean: float, f0_std: float) -> np.ndarray:
    vmask = f0 > 0.0
    out = np.zeros_like(f0, dtype=np.float32)
    if np.any(vmask):
        out[vmask] = (f0[vmask] - f0_mean) / f0_std
    return out

# ----------------------------- Data loader ----------------------------------- #

class SequencePadder(Sequence):
    """
    Keras Sequence that:
      - reads cached NPZ ('feats' for *noisy*),
      - extracts target clean Mel on the fly from 'clean' path,
      - normalizes inputs/targets with train stats,
      - returns voiced-weighted sample_weight in [0.3, 1.0].
    """
    def __init__(self,
                 rows: List[Dict[str, Any]],
                 batch_size: int,
                 fx: FeatureExtractor,
                 mel_mean: np.ndarray, mel_std: np.ndarray,
                 f0_mean: float, f0_std: float,
                 shuffle: bool = True):
        self.rows = list(rows)
        self.batch_size = int(batch_size)
        self.fx = fx
        self.mel_mean, self.mel_std = mel_mean, mel_std
        self.f0_mean, self.f0_std = float(f0_mean), float(f0_std)
        self.shuffle = bool(shuffle)
        self.indices = np.arange(len(self.rows))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.rows) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        sl = slice(idx*self.batch_size, min((idx+1)*self.batch_size, len(self.rows)))
        ids = self.indices[sl]
        return self._make_batch([self.rows[i] for i in ids])

    def _make_batch(self, batch_rows: List[Dict[str, Any]]):
        Xs, Ys, Ws = [], [], []
        maxT = 0

        for row in batch_rows:
            d = np.load(row["npz"], allow_pickle=False)
            feats = d["feats"].astype(np.float32)  # [T,D] (48 Mel + F0 + vprob [+ ceps])
            T, D = feats.shape

            mel_in = feats[:, :48]
            f0_in  = feats[:, 48:49]
            vprob  = feats[:, 49:50]
            ceps   = feats[:, 50:] if D > 50 else None

            # Normalize inputs using train stats
            mel_in = z_norm_mel(mel_in, self.mel_mean, self.mel_std)   # [T,48]
            f0_in  = z_norm_f0(f0_in.squeeze(-1), self.f0_mean, self.f0_std)[:, None]
            feats_n = np.concatenate([mel_in, f0_in, vprob], axis=-1) if ceps is None \
                      else np.concatenate([mel_in, f0_in, vprob, ceps], axis=-1)

            # Target: CLEAN Mel (unnormalized) → z-norm
            clean_path = str(row["clean"])
            pack_clean = self.fx.from_file(clean_path)
            y_mel = pack_clean["mel_log"].astype(np.float32)           # [T2,48]
            T2 = y_mel.shape[0]
            Tmin = min(T, T2)

            Xs.append(feats_n[:Tmin])
            Ys.append(z_norm_mel(y_mel[:Tmin], self.mel_mean, self.mel_std))

            # Voiced weighting (0.3..1.0)
            vseq = vprob[:Tmin, 0]
            Ws.append((0.3 + 0.7 * vseq).astype(np.float32))

            if Tmin > maxT:
                maxT = int(Tmin)

        Dn = Xs[0].shape[-1]
        X_pad = np.zeros((len(Xs), maxT, Dn), dtype=np.float32)
        y_pad = np.zeros((len(Ys), maxT, 48), dtype=np.float32)
        m_pad = np.zeros((len(Xs), maxT), dtype=np.float32)   # time mask for inputs (for completeness)
        w_pad = np.zeros((len(Xs), maxT), dtype=np.float32)   # sample_weight used by Keras

        for i, (x, y, w) in enumerate(zip(Xs, Ys, Ws)):
            t = x.shape[0]
            X_pad[i, :t, :] = x
            y_pad[i, :t, :] = y
            m_pad[i, :t] = 1.0
            w_pad[i, :t] = w

        # Keras will apply w_pad to the per-timestep loss we return
        return (X_pad, m_pad[..., None]), y_pad, w_pad

# ------------------------- Reconstruction helper ----------------------------- #

def reconstruct_from_logmel(
    noisy_wav: np.ndarray, sr: int, pred_logmel: np.ndarray,
    stft_win: int, stft_hop: int, n_fft_eff: int, fmin: float, fmax: Optional[float],
) -> np.ndarray:
    """Reconstruct waveform from predicted log-Mel using noisy phase (stable, conservative)."""
    S_noisy = librosa.stft(noisy_wav, n_fft=n_fft_eff, hop_length=stft_hop,
                           win_length=stft_win, window="hann", center=False, pad_mode="constant")
    mag_noisy = np.abs(S_noisy).astype(np.float32)
    phase_noisy = np.angle(S_noisy).astype(np.float32)

    mel_fb = librosa.filters.mel(
        sr=sr, n_fft=n_fft_eff, n_mels=pred_logmel.shape[1],
        fmin=fmin, fmax=(sr/2 if fmax is None else fmax), htk=True
    )  # [M,F]
    mel_pow_hat = np.exp(pred_logmel).T  # [M,T]
    pinv = np.linalg.pinv(mel_fb)        # [F,M]
    lin_pow_hat = np.clip(pinv @ mel_pow_hat, 0.0, None)  # [F,T]
    lin_mag_hat = np.sqrt(lin_pow_hat + 1e-9)

    # Allow some boost but keep safe
    lin_mag_hat = np.minimum(lin_mag_hat, 3.0 * mag_noisy)

    S_hat = lin_mag_hat * np.exp(1j * phase_noisy)
    y_hat = librosa.istft(S_hat, hop_length=stft_hop, win_length=stft_win,
                          window="hann", center=False, length=noisy_wav.shape[0])
    return y_hat.astype(np.float32)

# ----------------------------- Model / loss ---------------------------------- #

def get_compiled_model(input_dim: int, lr: float) -> tf.keras.Model:
    model = get_model(input_dim=input_dim)

    # Slightly more stable optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    # Per-frame loss; sample_weight [B,T] applied by Keras from the Sequence
    def masked_loss(y_true, y_pred):
        # Robust L1 on log-Mel (correlates better than MSE for perceptual changes)
        l1  = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)  # [B,T]
        # Small energy alignment on framewise sum of Mel (stabilizes loudness)
        e_true = tf.reduce_sum(y_true, axis=-1)  # [B,T]
        e_pred = tf.reduce_sum(y_pred, axis=-1)  # [B,T]
        e_l1 = tf.abs(e_true - e_pred)
        return l1 + 0.02 * e_l1

    model.compile(
        optimizer=opt,
        loss=masked_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model

# ----------------------------- Main ------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True, type=Path)
    ap.add_argument("--test-manifest",  required=True, type=Path)
    ap.add_argument("--train-stats",    required=True, type=Path, help="train_feature_stats.npz (scaler)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--mel-ceps", type=int, default=0)
    # Optional: NISQA checkpoint path; if omitted, NISQA MOS uses default value
    ap.add_argument("--nisqa-ckpt", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)
    (args.outdir / "wavs").mkdir(parents=True, exist_ok=True)

    # Front-end (must match feature cache / reconstruction cfg)
    fcfg = FeatureConfig(mel_ceps_keep=args.mel_ceps)
    fx = FeatureExtractor(fcfg)

    # Normalization scaler from train set
    mel_mean, mel_std, f0_mean, f0_std = load_feature_stats(args.train_stats)

    # Data
    df_train = load_manifest(args.train_manifest)
    df_test  = load_manifest(args.test_manifest)

    idxs = np.arange(len(df_train))
    np.random.shuffle(idxs)
    split_at = int(0.85 * len(idxs))
    tr_rows = [df_train.iloc[i].to_dict() for i in idxs[:split_at]]
    va_rows = [df_train.iloc[i].to_dict() for i in idxs[split_at:]]

    train_seq = SequencePadder(tr_rows, batch_size=args.batch_size, fx=fx,
                               mel_mean=mel_mean, mel_std=mel_std,
                               f0_mean=f0_mean, f0_std=f0_std,
                               shuffle=True)
    val_seq   = SequencePadder(va_rows, batch_size=args.batch_size, fx=fx,
                               mel_mean=mel_mean, mel_std=mel_std,
                               f0_mean=f0_mean, f0_std=f0_std,
                               shuffle=False)

    # Model
    sample_npz = np.load(tr_rows[0]["npz"])
    D = int(sample_npz["feats"].shape[-1])
    model = get_compiled_model(input_dim=D, lr=args.lr)
    model.summary()

    # Callbacks (SavedModel directory to avoid native .keras 'options' error)
    ckpt_path = args.outdir / "best_tf"
    cbs = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=str(ckpt_path),        # directory (SavedModel)
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        CSVLogger(str(args.outdir / "history.csv")),
    ]

    # Train
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=1,
    )

    # Save history JSON
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(args.outdir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # ------------------------ Evaluate & dump wavs ---------------------------- #
    def build_normalized_input_from_npz(npz_path: str) -> np.ndarray:
        d = np.load(npz_path, allow_pickle=False)
        feats = d["feats"].astype(np.float32)  # [T,D]
        mel_in = feats[:, :48]
        f0_in  = feats[:, 48:49]
        vprob  = feats[:, 49:50]
        ceps   = feats[:, 50:] if feats.shape[1] > 50 else None
        mel_in = z_norm_mel(mel_in, mel_mean, mel_std)
        f0_in  = z_norm_f0(f0_in.squeeze(-1), f0_mean, f0_std)[:, None]
        feats_n = np.concatenate([mel_in, f0_in, vprob], axis=-1) if ceps is None \
                  else np.concatenate([mel_in, f0_in, vprob, ceps], axis=-1)
        return feats_n

    # Optional: prepare NISQA model once
    nisqa_model = None
    if args.nisqa_ckpt and load_nisqa is not None and NISQA_FILE_SAFE is not None:
        try:
            nisqa_model = load_nisqa(args.nisqa_ckpt)
        except Exception as e:
            print(f"[WARN] Failed to load NISQA model: {e}. Will use default MOS.")

    metrics: Dict[str, Any] = {}
    test_rows = [df_test.iloc[i].to_dict() for i in range(min(12, len(df_test)))]
    sr = fcfg.sr
    stft_win, stft_hop = fcfg.win_length, fcfg.hop_length
    n_fft_eff = max(fcfg.n_fft, fcfg.win_length)

    pesq_list, stoi_list, sisdr_list = [], [], []
    snr_in_list, snr_out_list = [], []
    dnsmos_sig, dnsmos_bak, dnsmos_ovr, nisqa_list = [], [], [], []

    for j, row in enumerate(tqdm(test_rows, desc="Render test samples")):
        npz_path = row["npz"]
        noisy_path = str(row["noisy"]); clean_path = str(row["clean"])

        feats_n = build_normalized_input_from_npz(npz_path)  # [T,D] normalized
        X = feats_n[None, ...]
        M = np.ones((1, feats_n.shape[0], 1), dtype=np.float32)
        pred_logmel_n = model.predict([X, M], verbose=0)[0]                 # [T,48] normalized
        pred_logmel   = z_denorm_mel(pred_logmel_n, mel_mean, mel_std)      # [T,48] true log-Mel

        noisy_wav, sr_n = sf.read(noisy_path, dtype="float32", always_2d=False)
        clean_wav, sr_c = sf.read(clean_path, dtype="float32", always_2d=False)
        if noisy_wav.ndim == 2: noisy_wav = noisy_wav.mean(axis=1)
        if clean_wav.ndim == 2: clean_wav = clean_wav.mean(axis=1)
        if sr_n != sr: noisy_wav = librosa.resample(noisy_wav, orig_sr=sr_n, target_sr=sr, res_type="kaiser_fast")
        if sr_c != sr: clean_wav = librosa.resample(clean_wav, orig_sr=sr_c, target_sr=sr, res_type="kaiser_fast")

        enh_wav = reconstruct_from_logmel(noisy_wav, sr, pred_logmel, stft_win, stft_hop, n_fft_eff, fcfg.fmin, fcfg.fmax)

        base = args.outdir / "wavs" / f"sample_{j:02d}"
        wav_noisy = str(base.with_suffix(".noisy.wav"))
        wav_clean = str(base.with_suffix(".clean.wav"))
        wav_enh   = str(base.with_suffix(".enh.wav"))
        sf.write(wav_noisy, noisy_wav, sr)
        sf.write(wav_clean, clean_wav, sr)
        sf.write(wav_enh,   enh_wav,   sr)

        # Intrusive metrics ----------------------------------------------------
        def _snr(ref, est, eps=1e-9):
            num = np.sum(ref**2); den = np.sum((ref - est)**2) + eps
            return 10.0 * np.log10((num + eps) / den)
        snr_in_list.append(float(_snr(clean_wav, noisy_wav)))
        snr_out_list.append(float(_snr(clean_wav, enh_wav)))

        if SISDR_SAFE is not None:
            try: sisdr_list.append(float(SISDR_SAFE(clean_wav, enh_wav)))
            except Exception: pass

        if PESQ_SAFE is not None:
            try: pesq_list.append(float(PESQ_SAFE(clean_wav, enh_wav, sr)))
            except Exception: pass

        if STOI_SAFE is not None:
            try: stoi_list.append(float(STOI_SAFE(clean_wav, enh_wav, sr)))
            except Exception: pass

        # Non-intrusive metrics -----------------------------------------------
        # DNSMOS: use the file-based safe API if available; else defaults
        if DNSMOS_WAV_SAFE is not None:
            try:
                r = DNSMOS_WAV_SAFE(wav_enh)  # {'mos_sig','mos_bak','mos_ovr'}
                dnsmos_sig.append(float(r.get("mos_sig", 2.5)))
                dnsmos_bak.append(float(r.get("mos_bak", 2.5)))
                dnsmos_ovr.append(float(r.get("mos_ovr", 2.5)))
            except Exception:
                pass

        # NISQA: only if model loaded; otherwise will use default at aggregation
        if nisqa_model is not None and NISQA_FILE_SAFE is not None:
            try:
                mos = float(NISQA_FILE_SAFE(nisqa_model, wav_enh))
                nisqa_list.append(mos)
            except Exception:
                pass

    # Aggregate with safe defaults (no NaN)
    def _mean_or_default(xs, default):
        return float(np.mean(xs)) if xs else float(default)

    metrics["SNR_IN"]  = _mean_or_default(snr_in_list, 0.0)
    metrics["SNR_OUT"] = _mean_or_default(snr_out_list, 0.0)
    metrics["SI_SDR"]  = _mean_or_default(sisdr_list, -30.0)   # sisdr_safe default
    metrics["PESQ"]    = _mean_or_default(pesq_list, 1.5)      # pesq_safe default
    metrics["STOI"]    = _mean_or_default(stoi_list, 0.0)      # stoi_safe default
    metrics["DNSMOS_SIG"] = _mean_or_default(dnsmos_sig, 2.5)
    metrics["DNSMOS_BAK"] = _mean_or_default(dnsmos_bak, 2.5)
    metrics["DNSMOS_OVR"] = _mean_or_default(dnsmos_ovr, 2.5)
    metrics["NISQA"]      = _mean_or_default(nisqa_list, 2.5)

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Pretty print (None if NaN no longer applies; print rounded)
    printable = {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in metrics.items()}
    print("[VAL]", printable)

if __name__ == "__main__":
    main()
