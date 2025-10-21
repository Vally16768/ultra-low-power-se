#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py (TensorFlow)
Antrenare UNet1D (stil Mamba-UNet simplificat) pentru Speech Enhancement.

- Citește manifestul (clean_path, noisy_path, type[, needs_aug, aug_policy])
- Aplică augmentările DOAR unde e marcat:
    * dacă needs_aug==1 sau type=="augment":
         -> dacă aug_policy != "none": aplică lanț din augment/*
         -> altfel: mix cu zgomote "pure" din --noise_dir la SNR random
    * altfel (pre-paired & needs_aug==0): folosește noisy_path

- Progres:
    * tqdm la inițializarea pipeline-urilor (scanări / verificări)
    * progress bar Keras la antrenare (steps_per_epoch + validation_steps)

- Callbacks:
    * ReduceLROnPlateau
    * EarlyStopping (restore_best_weights)
    * ModelCheckpoint (best)

- Salvare:
    * SavedModel: {save_dir}/model_saved/
    * Keras H5:   {save_dir}/model.h5
    * Checkpoint best: {save_dir}/ckpt/best.keras
    * history.csv + history.json în {save_dir}/logs
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# === FIX: asigurăm că rădăcina proiectului e în sys.path pentru importuri locale ===
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent  # presupunem structura: <root>/{core,augment,...}
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

# Audio I/O + resampling
import soundfile as sf
try:
    import scipy.signal as sps
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import augment registry (numpy) — IMPORTANT: să existe augment/base.py etc.
from augment.base import apply_chain  # Registry se folosește intern via import-uri
import augment.babble as _aug_babble      # noqa: F401 (populate registry)
import augment.bandlimited as _aug_band   # noqa: F401
import augment.bursts as _aug_bursts      # noqa: F401
import augment.channel as _aug_channel    # noqa: F401
import augment.colored as _aug_colored    # noqa: F401
import augment.hum as _aug_hum            # noqa: F401
import augment.reverb as _aug_reverb      # noqa: F401


# ===========================
# Utilitare generale
# ===========================
AUTOTUNE = tf.data.AUTOTUNE

def set_all_seeds(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def print_once(msg: str):
    print(msg, flush=True)

# ===========================
# Audio helpers
# ===========================
def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    if _HAS_SCIPY:
        # polyphase resample
        g = np.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        return sps.resample_poly(x, up, down).astype(np.float32, copy=False)
    if _HAS_LIBROSA:
        return librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32, copy=False)
    raise RuntimeError(f"Nu pot resampla {sr}->{target_sr}: instalează scipy sau librosa.")

def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = _resample(x, sr, target_sr)
    return x.astype(np.float32, copy=False)

def pad_or_random_crop(x: np.ndarray, segment_len: int, rng: np.random.Generator) -> np.ndarray:
    L = x.shape[0]
    if L == segment_len:
        return x
    if L > segment_len:
        start = int(rng.integers(0, L - segment_len + 1))
        return x[start:start + segment_len]
    # pad
    out = np.zeros((segment_len,), dtype=np.float32)
    out[:L] = x
    return out

def add_noise_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    # aliniează lungimile
    Lc, Ln = clean.shape[0], noise.shape[0]
    if Ln >= Lc:
        start = int(rng.integers(0, Ln - Lc + 1))
        noise = noise[start:start + Lc]
    else:
        reps = 1 + (Lc // max(Ln, 1))
        noise = np.tile(noise, reps)[:Lc]

    p_clean = float(np.mean(clean**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    snr_lin = 10 ** (snr_db / 10.0)
    scale = np.sqrt(p_clean / (snr_lin * p_noise))
    noisy = clean + noise * scale
    maxv = np.max(np.abs(noisy))
    if maxv > 1.0:
        noisy = noisy / maxv
    return noisy.astype(np.float32, copy=False)

def si_snr_tf(y_pred: tf.Tensor, y_true: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Scale-Invariant SNR (dB), pe forme [B, T, 1].
    """
    y_true_z = y_true - tf.reduce_mean(y_true, axis=1, keepdims=True)
    y_pred_z = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    dot = tf.reduce_sum(y_pred_z * y_true_z, axis=1, keepdims=True)
    denom = tf.reduce_sum(y_true_z * y_true_z, axis=1, keepdims=True) + eps
    s_target = dot / denom * y_true_z
    e_noise = y_pred_z - s_target
    num = tf.reduce_sum(tf.square(s_target), axis=1) + eps
    den = tf.reduce_sum(tf.square(e_noise), axis=1) + eps
    ratio = num / den
    return 10.0 * tf.math.log(ratio) / tf.math.log(10.0)

# ===========================
# Augment policy → chain
# ===========================
def build_chain_from_policy(policy: str) -> List[Dict[str, Any]]:
    if not policy or policy == "none":
        return []
    p = str(policy).lower()
    if p.startswith("balance:"):
        return [
            {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 10.0}},
            {"name": "reverb_toy", "params": {"rt60": 0.2}},
        ]
    if p.startswith("harder:"):
        return [
            {"name": "reverb_toy", "params": {"rt60": 0.5}},
            {"name": "bandlimit_noise", "params": {"lo": 300, "hi": 3200, "snr_db": 8.0}},
            {"name": "add_colored_noise", "params": {"color": "white", "snr_db": 15.0}},
        ]
    return [
        {"name": "reverb_toy", "params": {"rt60": 0.3}},
        {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 12.0}},
    ]

# ===========================
# Model (UNet1D simplificat, ONNX-friendly)
# ===========================
from tensorflow.keras import layers

def conv_block(x, ch, k=9, s=1):
    x = layers.Conv1D(ch, k, s, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def down_block(x, ch):
    c1 = conv_block(x, ch)
    c2 = conv_block(c1, ch)
    p  = layers.AveragePooling1D(2)(c2)
    return c2, p

def up_block(x, skip, ch):
    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, ch)
    x = conv_block(x, ch)
    return x

def build_unet1d(input_len: Optional[int], base_ch: int = 64) -> keras.Model:
    """
    UNet1D compatibil Keras → SavedModel (ușor de convertit ONNX).
    Intrare: (T,1). Ieșire: (T,1).
    """
    inp = keras.Input(shape=(input_len, 1), name="noisy_in")
    # Encoder
    s1, p1 = down_block(inp, base_ch)
    s2, p2 = down_block(p1,  base_ch*2)
    s3, p3 = down_block(p2,  base_ch*4)
    b      = conv_block(p3,  base_ch*8)
    # Decoder
    u3 = up_block(b,  s3, base_ch*4)
    u2 = up_block(u3, s2, base_ch*2)
    u1 = up_block(u2, s1, base_ch)
    out = layers.Conv1D(1, 1, padding="same", name="enhanced")(u1)
    return keras.Model(inp, out, name="UNet1D_SE")

# ===========================
# tf.data generator din manifest (cu augment condiționat)
# ===========================
@dataclass
class DSConfig:
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    min_snr_db: float = -5.0
    max_snr_db: float = 20.0
    force_chain: bool = False

from typing import Iterable

def make_generator(
    df: pd.DataFrame,
    cfg: DSConfig,
    noise_files: List[Path],
    babble_pool_paths: List[str],
    shuffle: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)

    def _apply_chain(clean: np.ndarray, chain: List[Dict[str, Any]]) -> np.ndarray:
        # completează pool_paths pentru add_babble dacă lipsesc
        c2 = []
        for st in chain:
            s = dict(st)
            params = dict(s.get("params", {}))
            if s.get("name") == "add_babble" and "pool_paths" not in params and babble_pool_paths:
                params["pool_paths"] = babble_pool_paths
            s["params"] = params
            c2.append(s)
        return apply_chain(clean, sr=cfg.sample_rate, chain=c2, rng=rng).astype(np.float32, copy=False)

    def _mix_noise(clean: np.ndarray) -> np.ndarray:
        if not noise_files:
            return clean.copy()
        npath = str(noise_files[int(rng.integers(0, len(noise_files)))])
        noise = load_audio_mono(npath, cfg.sample_rate)
        snr_db = float(rng.uniform(cfg.min_snr_db, cfg.max_snr_db))
        return add_noise_snr(clean, noise, snr_db, rng)

    idxs = np.arange(len(df))
    while True:
        if shuffle:
            rng.shuffle(idxs)
        for i in idxs:
            row = df.iloc[i]
            typ = str(row.get("type", "pre-paired"))
            needs_aug = str(row.get("needs_aug", "0")).lower() in {"1", "true", "yes"}
            policy = str(row.get("aug_policy", "none"))

            clean = load_audio_mono(str(row["clean_path"]), cfg.sample_rate)
            T = int(cfg.segment_seconds * cfg.sample_rate)
            clean = pad_or_random_crop(clean, T, rng)

            if typ == "pre-paired" and not needs_aug:
                noisy = load_audio_mono(str(row["noisy_path"]), cfg.sample_rate)
                noisy = pad_or_random_crop(noisy, T, rng)
            else:
                chain = build_chain_from_policy(policy)
                use_chain = (cfg.force_chain and len(chain) > 0) or (policy != "none" and len(chain) > 0)
                if use_chain:
                    noisy = _apply_chain(clean, chain)
                else:
                    noisy = _mix_noise(clean)

            # [T] -> [T,1]
            yield noisy[:, None], clean[:, None]

def build_tf_dataset(
    csv_path: str,
    cfg: DSConfig,
    noise_dir: Optional[str],
    babble_pool_dir: Optional[str],
    batch_size: int,
    shuffle: bool,
    seed: int,
    take_steps: Optional[int] = None,
):
    df = pd.read_csv(csv_path)
    # Scanări cu progres
    print_once(f"[Info] {csv_path}: {len(df)} rânduri")
    noise_files: List[Path] = []
    if noise_dir:
        print_once(f"[Scan] zgomote (noise_dir): {noise_dir}")
        noise_files = list(tqdm(Path(noise_dir).rglob("*.wav"), desc="  zgomote", unit="wav"))
    babble_pool_paths: List[str] = []
    if babble_pool_dir:
        print_once(f"[Scan] babble pool: {babble_pool_dir}")
        babble_pool_paths = [str(p) for p in tqdm(Path(babble_pool_dir).rglob("*.wav"), desc="  babble", unit="wav")]

    gen = make_generator(
        df=df,
        cfg=cfg,
        noise_files=noise_files,
        babble_pool_paths=babble_pool_paths,
        shuffle=shuffle,
        seed=seed,
    )

    output_sig = (
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(lambda: gen, output_signature=output_sig)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(df)

# ===========================
# History helpers
# ===========================
@dataclass
class HistRow:
    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    val_si_snr: Optional[float] = None

def save_history(save_dir: Path, history: keras.callbacks.History, extra: Dict[str, Any]):
    logs_dir = save_dir / "logs"
    ensure_dir(logs_dir)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(logs_dir / "history.csv", index=False)

    summary = {
        "epochs": len(hist_df),
        **extra,
        "final": {k: float(hist_df[k].iloc[-1]) for k in hist_df.columns if len(hist_df[k]) > 0},
    }
    with open(logs_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)

# ===========================
# Main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=str)
    ap.add_argument("--val_csv",   required=True, type=str)
    ap.add_argument("--save_dir",  default="artifacts/tf_run1", type=str)
    ap.add_argument("--noise_dir", default=None, type=str, help="Director cu zgomote PURE pentru mix SNR")
    ap.add_argument("--babble_pool_dir", default=None, type=str, help="WAV-uri pt add_babble (dacă apare în policy)")

    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--sample_rate", default=16000, type=int)
    ap.add_argument("--segment_seconds", default=2.0, type=float)
    ap.add_argument("--min_snr_db", default=-5.0, type=float)
    ap.add_argument("--max_snr_db", default=20.0, type=float)
    ap.add_argument("--seed", default=123, type=int)

    # model
    ap.add_argument("--base_ch", default=64, type=int)
    ap.add_argument("--input_len", default=None, type=int, help="Dacă e None, modelul acceptă T variabil (recomandat)")

    # callbacks
    ap.add_argument("--patience_es", default=7, type=int, help="EarlyStopping patience")
    ap.add_argument("--patience_rlr", default=3, type=int, help="ReduceLROnPlateau patience")
    ap.add_argument("--rlr_factor", default=0.5, type=float)

    args = ap.parse_args()
    set_all_seeds(args.seed)

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    ensure_dir(save_dir / "ckpt")
    ensure_dir(save_dir / "logs")

    # ===========================
    # Datasets (tf.data)
    # ===========================
    cfg = DSConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        min_snr_db=args.min_snr_db,
        max_snr_db=args.max_snr_db,
        force_chain=False,  # nu forțăm chain pe val
    )

    print_once("[Step] Construiesc pipeline-ul de training...")
    train_ds, n_train = build_tf_dataset(
        csv_path=args.train_csv,
        cfg=cfg,
        noise_dir=args.noise_dir,
        babble_pool_dir=args.babble_pool_dir,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    steps_per_epoch = max(1, n_train // args.batch_size)
    print_once(f"[Info] steps_per_epoch: {steps_per_epoch}")

    print_once("[Step] Construiesc pipeline-ul de validare...")
    val_cfg = DSConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        min_snr_db=args.min_snr_db,
        max_snr_db=args.max_snr_db,
        force_chain=False,
    )
    val_ds, n_val = build_tf_dataset(
        csv_path=args.val_csv,
        cfg=val_cfg,
        noise_dir=None,  # NU mixăm zgomot pe validare
        babble_pool_dir=None,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
    )
    validation_steps = max(1, n_val // args.batch_size)
    print_once(f"[Info] validation_steps: {validation_steps}")

    # ===========================
    # Model + compile
    # ===========================
    model = build_unet1d(
        input_len=args.input_len,  # None => T variabil
        base_ch=args.base_ch
    )
    model.summary(print_fn=lambda s: print_once(s))

    # Loss + metrics
    loss = keras.losses.MeanAbsoluteError()
    metrics = [
        keras.metrics.MeanAbsoluteError(name="mae"),
        si_snr_tf,
    ]
    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)

    # ===========================
    # Callbacks (ES + ReduceLR + CKPT)
    # ===========================
    cbs = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "ckpt" / "best.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.rlr_factor,
            patience=args.patience_rlr,
            min_delta=1e-4,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(save_dir / "logs" / "history.csv"), append=False),
    ]

    # ===========================
    # Train (progres Keras)
    # ===========================
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1,
    )

    # ===========================
    # Save final models (ONNX-friendly)
    # ===========================
    # 1) SavedModel (preferat pentru onnx conversion)
    savedmodel_dir = save_dir / "model_saved"
    model.save(savedmodel_dir, include_optimizer=False)
    # 2) H5 (opțional)
    model.save(save_dir / "model.h5", include_optimizer=False)

    # ===========================
    # Save history (JSON summary)
    # ===========================
    # CSV e deja scris de CSVLogger; adăugăm JSON cu extra info
    extra = {
        "args": vars(args),
        "n_train_rows": n_train,
        "n_val_rows": n_val,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }
    hist_df = pd.read_csv(save_dir / "logs" / "history.csv")
    summary = {
        **extra,
        "final": {k: float(hist_df[k].iloc[-1]) for k in hist_df.columns if len(hist_df[k]) > 0},
    }
    with open(save_dir / "logs" / "history.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_once(f"[OK] Model salvat: {savedmodel_dir}")
    print_once(f"[OK] Istoric antrenare: {save_dir/'logs'/'history.csv'} & {save_dir/'logs'/'history.json'}")


if __name__ == "__main__":
    # Optimize TF thread pools automat
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "0")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "0")
    main()
