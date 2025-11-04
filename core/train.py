#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/train.py — Track 1 training:
- Loads cached NOISY features (from your feature cache)
- Target: CLEAN log-Mel computed on-the-fly with FeatureExtractor
- Split: 85% train / 15% val from the train manifest
- Model: core.models.model.get_model(input_dim)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
- Saves: metrics.json, history.csv/json, best.h5, and a few (noisy/clean/enhanced) wavs
"""

from __future__ import annotations
import argparse, json, os, random, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

# Local imports
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.features.feature import FeatureConfig, FeatureExtractor

try:
    from metrics.pesq import pesq_score_safe as PESQ
except Exception:
    PESQ = None
try:
    from metrics.stoi import stoi_score_safe as STOI
except Exception:
    STOI = None


# ----------------------------- Utils ----------------------------------------- #

def set_seed(seed: int = 41):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def reconstruct_from_logmel(
    noisy_wav: np.ndarray, sr: int, pred_logmel: np.ndarray,
    stft_win: int, stft_hop: int, n_fft_eff: int, fmin: float, fmax: Optional[float],
) -> np.ndarray:
    S_noisy = librosa.stft(noisy_wav, n_fft=n_fft_eff, hop_length=stft_hop,
                           win_length=stft_win, window="hann", center=False, pad_mode="constant")
    mag_noisy = np.abs(S_noisy).astype(np.float32)
    phase_noisy = np.angle(S_noisy).astype(np.float32)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft_eff, n_mels=pred_logmel.shape[1],
                                 fmin=fmin, fmax=(sr/2 if fmax is None else fmax), htk=True)
    mel_pow_hat = np.exp(pred_logmel).T  # [M,T]
    pinv = np.linalg.pinv(mel_fb)        # [F,M]
    lin_pow_hat = np.clip(pinv @ mel_pow_hat, 0.0, None)  # [F,T]
    lin_mag_hat = np.sqrt(lin_pow_hat + 1e-9)
    lin_mag_hat = np.minimum(lin_mag_hat, 2.0 * mag_noisy)
    S_hat = lin_mag_hat * np.exp(1j * phase_noisy)
    y_hat = librosa.istft(S_hat, hop_length=stft_hop, win_length=stft_win,
                          window="hann", center=False, length=noisy_wav.shape[0])
    return y_hat.astype(np.float32)


# ----------------------------- Data ------------------------------------------ #

class SequencePadder(tf.keras.utils.Sequence):
    """
    Keras Sequence that:
      - reads per-utterance noisy features from NPZ (from the manifest)
      - computes per-utterance clean log-mel via FeatureExtractor
      - pads variable length to the batch max and returns sample_weight mask
    Inputs X:  [B, T, D] (feats: mel(48)+f0+vprob[+cepstra])
    Targets y: [B, T, 48] (clean log-Mel)
    sample_weight: [B, T] mask (1.0 for valid frames, 0.0 for padding)
    """
    def __init__(self, rows: List[Dict[str, Any]], batch_size: int,
                 fx: FeatureExtractor, shuffle: bool = True):
        self.rows = rows
        self.batch_size = batch_size
        self.fx = fx
        self.shuffle = shuffle
        self.idxs = np.arange(len(rows))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.idxs)

    def __getitem__(self, index: int):
        batch_ids = self.idxs[index * self.batch_size : (index + 1) * self.batch_size]
        Xs, Ys = [], []
        maxT = 0
        for bi in batch_ids:
            row = self.rows[bi]
            d = np.load(row["npz"], allow_pickle=False)
            feats = d["feats"].astype(np.float32)               # [T, D]
            clean_pack = self.fx.from_file(str(d["clean_path"]))
            y = clean_pack["mel_log"].astype(np.float32)        # [T, 48]
            T = min(len(feats), len(y))
            feats = feats[:T]; y = y[:T]
            Xs.append(feats); Ys.append(y)
            if T > maxT: maxT = T

        D = Xs[0].shape[-1]
        X_pad = np.zeros((len(Xs), maxT, D), dtype=np.float32)
        y_pad = np.zeros((len(Ys), maxT, Ys[0].shape[-1]), dtype=np.float32)
        m_pad = np.zeros((len(Xs), maxT), dtype=np.float32)  # [B, T]
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            T = x.shape[0]
            X_pad[i, :T, :] = x
            y_pad[i, :T, :] = y
            m_pad[i, :T] = 1.0
        # Keras expects sample_weight shape [B, T] for time-distributed losses
        return (X_pad, m_pad[..., None]), y_pad, m_pad


# ----------------------------- Model ----------------------------------------- #

def get_compiled_model(input_dim: int, lr: float) -> tf.keras.Model:
    from core.models.model import get_model
    model = get_model(input_dim=input_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Loss returns [B,T]; Keras will apply sample_weight [B,T]
    def masked_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)  # [B,T]
    model.compile(optimizer=opt, loss=masked_mse, metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model


# ----------------------------- Main ------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True, type=Path)
    ap.add_argument("--test-manifest",  required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--mel-ceps", type=int, default=24)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)
    (args.outdir / "wavs").mkdir(parents=True, exist_ok=True)

    fcfg = FeatureConfig(mel_ceps_keep=args.mel_ceps)
    fx = FeatureExtractor(fcfg)

    df_train = load_manifest(args.train_manifest)
    df_test  = load_manifest(args.test_manifest)

    idxs = np.arange(len(df_train))
    np.random.shuffle(idxs)
    split_at = int(0.85 * len(idxs))
    tr_rows = [df_train.iloc[i].to_dict() for i in idxs[:split_at]]
    va_rows = [df_train.iloc[i].to_dict() for i in idxs[split_at:]]

    train_seq = SequencePadder(tr_rows, batch_size=args.batch_size, fx=fx, shuffle=True)
    val_seq   = SequencePadder(va_rows, batch_size=args.batch_size, fx=fx, shuffle=False)

    sample_npz = np.load(tr_rows[0]["npz"])
    D = int(sample_npz["feats"].shape[-1])
    model = get_compiled_model(input_dim=D, lr=args.lr)
    model.summary()

    ckpt_path = args.outdir / "best.h5"
    cbs = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
        CSVLogger(str(args.outdir / "history.csv")),
    ]

    # >>>>>>>>>>>>>>>>>> TRAIN (with sample_weight from sequence) <<<<<<<<<<<<<< #
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=1,
    )
    # Keras automatically reads sample_weight from the sequence's third return value.

    # Save history
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(args.outdir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # ------------------------ Evaluate & dump wavs ---------------------------- #
    metrics: Dict[str, Any] = {}
    test_rows = [df_test.iloc[i].to_dict() for i in range(min(12, len(df_test)))]
    sr = fcfg.sr
    stft_win, stft_hop = fcfg.win_length, fcfg.hop_length
    n_fft_eff = max(fcfg.n_fft, fcfg.win_length)

    pesq_list, stoi_list, snr_in_list, snr_out_list = [], [], [], []

    for j, row in enumerate(tqdm(test_rows, desc="Render test samples")):
        d = np.load(row["npz"], allow_pickle=False)
        feats = d["feats"].astype(np.float32)
        noisy_path = str(d["noisy_path"]); clean_path = str(d["clean_path"])

        X = feats[None, ...]
        M = np.ones((1, feats.shape[0], 1), dtype=np.float32)
        pred_logmel = model.predict([X, M], verbose=0)[0]

        noisy_wav, sr_n = sf.read(noisy_path, dtype="float32", always_2d=False)
        clean_wav, sr_c = sf.read(clean_path, dtype="float32", always_2d=False)
        if noisy_wav.ndim == 2: noisy_wav = noisy_wav.mean(axis=1)
        if clean_wav.ndim == 2: clean_wav = clean_wav.mean(axis=1)
        if sr_n != sr: noisy_wav = librosa.resample(noisy_wav, orig_sr=sr_n, target_sr=sr, res_type="kaiser_fast")
        if sr_c != sr: clean_wav = librosa.resample(clean_wav, orig_sr=sr_c, target_sr=sr, res_type="kaiser_fast")

        enh_wav = reconstruct_from_logmel(noisy_wav, sr, pred_logmel, stft_win, stft_hop, n_fft_eff, fcfg.fmin, fcfg.fmax)

        base = args.outdir / "wavs" / f"sample_{j:02d}"
        sf.write(str(base.with_suffix(".noisy.wav")), noisy_wav, sr)
        sf.write(str(base.with_suffix(".clean.wav")), clean_wav, sr)
        sf.write(str(base.with_suffix(".enh.wav")),   enh_wav,   sr)

        def snr(ref, est, eps=1e-9):
            num = np.sum(ref**2); den = np.sum((ref - est)**2) + eps
            return 10.0 * np.log10((num + eps) / den)
        snr_in_list.append(float(snr(clean_wav, noisy_wav)))
        snr_out_list.append(float(snr(clean_wav, enh_wav)))

        if PESQ is not None:
            try: pesq_list.append(float(PESQ(clean_wav, enh_wav, sr)))
            except Exception: pass
        if STOI is not None:
            try: stoi_list.append(float(STOI(clean_wav, enh_wav, sr)))
            except Exception: pass

    metrics["snr_in_mean"]  = float(np.mean(snr_in_list)) if snr_in_list else None
    metrics["snr_out_mean"] = float(np.mean(snr_out_list)) if snr_out_list else None
    metrics["snr_delta"]    = (metrics["snr_out_mean"] - metrics["snr_in_mean"]) if (metrics["snr_in_mean"] is not None and metrics["snr_out_mean"] is not None) else None
    if pesq_list: metrics["pesq_mean"] = float(np.mean(pesq_list))
    if stoi_list: metrics["stoi_mean"] = float(np.mean(stoi_list))

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[RESULTS]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
