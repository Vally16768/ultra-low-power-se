#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — STRICT variant (no NISQA)

- Intrusive metrics: SI-SDR, PESQ, STOI, SNR (noisy/enhanced)
- Non-intrusive metric: DNSMOS (wav-level)
- Strict reconstruction (NO fallbacks):
    * Normalize noisy to feature domain peak
    * STFT/ISTFT use EXACT same centering as FE (or CLI if FE lacks it)
    * Invert log-Mel with EXACT base & epsilon from FE (or CLI)
    * Mel->linear via Tikhonov (ridge) in POWER domain
    * Cap + controlled noisy-mag blend in SAME domain
    * ISTFT back, then de-normalize to original amplitude
"""

from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
_ROOT = _THIS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Front-end / model
from core.features.feature import FeatureConfig, FeatureExtractor   # noqa: E402
from core.models.model import get_model                             # noqa: E402

# STRICT metrics
from metrics.snr import snr_noisy, snr_enhanced                     # noqa: E402
from metrics.sisdr import sisdr                                     # noqa: E402
from metrics.pesq import pesq_score                                 # noqa: E402
from metrics.stoi import stoi_score                                 # noqa: E402
from metrics.dnsmos import dnsmos_wav                               # noqa: E402


# ----------------------------- Utils ----------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _norm_path(path_str: str) -> str:
    """Normalize Windows-style paths when running under Linux/WSL."""
    return str(Path(str(path_str).replace("\\", "/")))

def load_feature_stats(stats_npz_path: Path):
    d = np.load(str(stats_npz_path))
    for k in ("mel_mean", "mel_std", "f0_mean", "f0_std"):
        if k not in d:
            raise KeyError(f"Missing '{k}' in {stats_npz_path}")
    mel_mean = d["mel_mean"].astype(np.float32)
    mel_std  = d["mel_std"].astype(np.float32)
    f0_mean  = float(d["f0_mean"])
    f0_std   = float(d["f0_std"])
    mel_std  = np.maximum(mel_std, 1e-6)
    f0_std   = max(f0_std, 1e-6)
    if mel_mean.shape[0] != 48 or mel_std.shape[0] != 48:
        raise ValueError("Expected 48 mel bins in statistics.")
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

def _assert_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite: {value}")

def _align_pair_strict(ref: np.ndarray, other: np.ndarray, name_a: str, name_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Hard crop both signals to the same min length; raise if empty."""
    L = min(ref.shape[0], other.shape[0])
    if L <= 0:
        raise ValueError(f"Cannot align {name_a} and {name_b}: non-positive min length "
                         f"(len({name_a})={ref.shape[0]}, len({name_b})={other.shape[0]}).")
    return ref[:L], other[:L]


# ----------------------------- Data loader ----------------------------------- #

class SequencePadder(Sequence):
    """Loads NPZ rows, normalizes with saved train stats, and pads batches (STRICT)."""
    def __init__(self, rows: List[Dict[str, Any]], batch_size: int, fx: FeatureExtractor,
                 mel_mean: np.ndarray, mel_std: np.ndarray, f0_mean: float, f0_std: float,
                 shuffle: bool, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.batch = int(batch_size)
        self.fx = fx
        self.mel_mean = mel_mean
        self.mel_std  = mel_std
        self.f0_mean  = f0_mean
        self.f0_std   = f0_std
        self.shuffle = bool(shuffle)
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.rows) + self.batch - 1) // self.batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.rows)

    def __getitem__(self, idx: int):
        rows = self.rows[idx*self.batch:(idx+1)*self.batch]
        Xs, Ys, Ws = [], [], []
        maxT = 0

        for row in rows:
            d = np.load(_norm_path(row["npz"]), allow_pickle=False)
            feats = d["feats"].astype(np.float32)  # [T, D] unnormalized: [mel_log(48), f0, vprob, (opt) ceps]
            T, D = feats.shape

            if D < 50:
                raise ValueError(f"Expected at least 50 dims ([48 mel]+f0+vprob), got {D}")

            mel_in = feats[:, :48]
            f0_in  = feats[:, 48:49]
            vprob  = feats[:, 49:50]
            ceps   = feats[:, 50:] if D > 50 else None

            # Inputs use z-norm (STRICT)
            mel_in = z_norm_mel(mel_in, self.mel_mean, self.mel_std)
            f0_in  = z_norm_f0(f0_in.squeeze(-1), self.f0_mean, self.f0_std)[:, None]
            feats_n = np.concatenate([mel_in, f0_in, vprob], axis=-1) if ceps is None \
                      else np.concatenate([mel_in, f0_in, vprob, ceps], axis=-1)

            # Targets: CLEAN Mel (unnormalized) -> z-norm (STRICT)
            clean_path = _norm_path(str(row["clean"]))
            pack_clean = self.fx.from_file(clean_path)
            if "mel_log" not in pack_clean:
                raise KeyError("FeatureExtractor must return 'mel_log' for clean.")
            y_mel = pack_clean["mel_log"].astype(np.float32)           # [T2,48]
            if y_mel.shape[1] != 48:
                raise ValueError("Expected 48 mel bins in 'mel_log'.")

            T2 = y_mel.shape[0]
            Tmin = min(T, T2)

            Xs.append(feats_n[:Tmin])
            Ys.append(z_norm_mel(y_mel[:Tmin], self.mel_mean, self.mel_std))

            # Voiced weighting (STRICT)
            vseq = vprob[:Tmin, 0]
            Ws.append((0.3 + 0.7 * vseq).astype(np.float32))

            if Tmin > maxT:
                maxT = int(Tmin)

        Dn = Xs[0].shape[-1]
        X_pad = np.zeros((len(Xs), maxT, Dn), dtype=np.float32)
        y_pad = np.zeros((len(Ys), maxT, 48), dtype=np.float32)
        m_pad = np.zeros((len(Xs), maxT), dtype=np.float32)
        w_pad = np.zeros((len(Xs), maxT), dtype=np.float32)

        for i, (x, y, w) in enumerate(zip(Xs, Ys, Ws)):
            t = x.shape[0]
            X_pad[i, :t, :] = x
            y_pad[i, :t, :] = y
            m_pad[i, :t]    = 1.0
            w_pad[i, :t]    = w

        return (X_pad, m_pad[..., None]), y_pad, w_pad


# ------------------------- Reconstruction (STRICT domain) --------------------- #

def _peak_normalize(wav: np.ndarray, peak_target: float, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
    p = float(np.max(np.abs(wav)) + eps)
    scale = peak_target / p if p > 0 else 1.0
    return (wav * scale).astype(np.float32), scale

def reconstruct_from_logmel_strict(
    *,
    noisy_wav: np.ndarray,
    sr: int,
    pred_logmel: np.ndarray,      # [T, M] true log-Mel, NOT z-normed
    stft_win: int,
    stft_hop: int,
    n_fft_eff: int,
    mel_fb: np.ndarray,           # [M, F] EXACT filterbank used by FE
    log_base: str,                # "ln" or "log10" (REQUIRED)
    log_eps: float,               # epsilon used before log (REQUIRED)
    center: bool,                 # EXACT same as FE
    peak_target: float,           # EXACT same as FE
    cap_ratio: float,             # e.g., 6.0
    ridge: float,                 # e.g., 1e-3
    noisy_blend: float            # e.g., 0.20
) -> np.ndarray:
    noisy_wav = np.asarray(noisy_wav, dtype=np.float32).ravel()

    # 1) Normalize to feature domain
    noisy_norm, scale = _peak_normalize(noisy_wav, peak_target)

    # 2) STFT in EXACT same config as FE
    S_noisy = librosa.stft(
        noisy_norm,
        n_fft=n_fft_eff,
        hop_length=stft_hop,
        win_length=stft_win,
        window="hann",
        center=center,
        pad_mode=("reflect" if center else "constant"),
    )
    mag_noisy = np.abs(S_noisy).astype(np.float32)   # [F, Tn]
    phase_noisy = np.angle(S_noisy).astype(np.float32)

    # 3) Invert log-Mel using EXACT base + epsilon
    if log_base.lower() in ("ln", "e", "natural"):
        mel_pow_hat = np.exp(pred_logmel) - float(log_eps)
    elif log_base.lower() in ("log10", "10"):
        mel_pow_hat = (10.0 ** pred_logmel) - float(log_eps)
    else:
        raise ValueError(f"Unsupported log_base: {log_base!r}")
    mel_pow_hat = np.maximum(mel_pow_hat, 0.0).astype(np.float32)  # [T̂, M]
    mel_pow_hat = mel_pow_hat.T                                    # -> [M, T̂]

    # 4) Tikhonov-regularized Mel->linear in POWER domain
    FB = mel_fb.astype(np.float32)                # [M, F]
    if FB.ndim != 2 or FB.shape[0] != mel_pow_hat.shape[0]:
        raise ValueError("mel_fb shape mismatch with predicted Mel.")
    F = FB.shape[1]
    AtA = FB.T @ FB
    AtA.flat[::F+1] += float(ridge)               # add λ to diagonal
    AtY = FB.T @ mel_pow_hat
    lin_pow_hat = np.linalg.solve(AtA, AtY)       # [F, T̂]
    lin_pow_hat = np.maximum(lin_pow_hat, 0.0)
    lin_mag_hat = np.sqrt(lin_pow_hat + 1e-12)    # [F, T̂]

    # -------- STRICT FRAME ALIGNMENT --------
    Tn = mag_noisy.shape[1]
    That = lin_mag_hat.shape[1]
    T = min(Tn, That)
    if T <= 0:
        raise ValueError(f"Non-positive frame intersection: Tn={Tn}, T̂={That}")

    # Trim BOTH streams to the same T (left-aligned)
    mag_noisy   = mag_noisy[:,   :T]
    phase_noisy = phase_noisy[:, :T]
    lin_mag_hat = lin_mag_hat[:, :T]
    # ----------------------------------------

    # 5) Cap + blend in SAME (normalized) domain
    if not (0.0 <= noisy_blend <= 1.0):
        raise ValueError("noisy_blend must be in [0,1].")
    lin_mag_hat = np.minimum(lin_mag_hat, float(cap_ratio) * mag_noisy)
    lin_mag_hat = (1.0 - noisy_blend) * lin_mag_hat + noisy_blend * mag_noisy

    # 6) ISTFT with exact centering; then de-normalize amplitude
    S_hat = lin_mag_hat * np.exp(1j * phase_noisy)

    # If center=False, make the time length consistent with the trimmed T
    istft_length = None
    if not center:
        # With center=False, n_frames ≈ 1 + (n_samples - n_fft) // hop  →  n_samples ≈ T*hop + n_fft
        istft_length = int(T * stft_hop + n_fft_eff)

    y_hat_norm = librosa.istft(
        S_hat,
        hop_length=stft_hop,
        win_length=stft_win,
        window="hann",
        center=center,
        length=istft_length,
    ).astype(np.float32)

    # For center=True, enforce same length as normalized noisy (trim only)
    if center and y_hat_norm.shape[0] != noisy_norm.shape[0]:
        L = min(y_hat_norm.shape[0], noisy_norm.shape[0])
        y_hat_norm = y_hat_norm[:L]

    y_hat = (y_hat_norm / max(scale, 1e-8)).astype(np.float32)
    return y_hat


# ----------------------------- Model / loss ---------------------------------- #

def get_compiled_model(
    input_dim: int,
    lr: float,
    mel_mean: np.ndarray,
    mel_std: np.ndarray,
    log_base: str,
    log_eps: float,
) -> tf.keras.Model:
    """
    Build and compile model with a mixed loss:
    - L1 on z-normalized log-Mel (stabilizing term)
    - spectral convergence and log-magnitude in true Mel-power domain
    - small energy consistency term
    """
    model = get_model(input_dim=input_dim)

    opt = tf.keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    # Convert stats & params to TF constants for use in the loss
    mel_mean_tf = tf.constant(mel_mean.reshape(1, 1, -1), dtype=tf.float32)  # [1,1,48]
    mel_std_tf  = tf.constant(mel_std.reshape(1, 1, -1),  dtype=tf.float32)  # [1,1,48]
    log_eps_tf  = tf.constant(float(log_eps), dtype=tf.float32)

    use_ln = (log_base.lower() in ("ln", "e", "natural"))

    # Loss mixt: L1 (z-norm) + spectral convergence + log-mag + energy, toate în domeniul corect
    def masked_loss(y_true, y_pred):
        """
        y_true, y_pred: [B, T, 48] log-Mel NORMALIZAT (z-norm).
        Return: [B, T] loss per-frame; Keras folosește sample_weight din SequencePadder.
        """

        # 0) De-normalizăm la log-Mel real
        logmel_true = y_true * mel_std_tf + mel_mean_tf   # [B,T,48]
        logmel_pred = y_pred * mel_std_tf + mel_mean_tf   # [B,T,48]

        # 1) L1 pe log-Mel NORMALIZAT (stabilizator)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)  # [B,T]

        # 2) Convertim log-Mel -> Mel-power în același mod ca la reconstrucție
        if use_ln:
            mel_pow_true = tf.exp(logmel_true) - log_eps_tf
            mel_pow_pred = tf.exp(logmel_pred) - log_eps_tf
        else:
            mel_pow_true = tf.pow(10.0, logmel_true) - log_eps_tf
            mel_pow_pred = tf.pow(10.0, logmel_pred) - log_eps_tf

        mel_pow_true = tf.nn.relu(mel_pow_true)
        mel_pow_pred = tf.nn.relu(mel_pow_pred)

        # 3) Spectral convergence în Mel-power
        #    sc = ||P_true - P_pred|| / (||P_true|| + eps), agregat pe frecvență
        num = tf.norm(mel_pow_true - mel_pow_pred, ord='euclidean', axis=-1)  # [B,T]
        den = tf.norm(mel_pow_true,              ord='euclidean', axis=-1) + 1e-8
        l_sc = num / den  # [B,T]

        # 4) Log-magnitude loss în Mel-power
        log_mag_true = tf.math.log(mel_pow_true + 1e-8)
        log_mag_pred = tf.math.log(mel_pow_pred + 1e-8)
        l_logmag = tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred), axis=-1)  # [B,T]

        # 5) Termen de energie (pe log-Mel real)
        e_true = tf.reduce_sum(logmel_true, axis=-1)  # [B, T]
        e_pred = tf.reduce_sum(logmel_pred, axis=-1)  # [B, T]
        e_l1 = tf.abs(e_true - e_pred)

        # 6) Combinația finală
        #    - 0.5 * log-mag + 0.5 * sc: termeni “perceptuali” principali
        #    - 0.2 * L1 z-norm: stabilizare
        #    - 0.01 * energy: mic, doar pentru consistență de energie
        loss = (
            0.2 * l1
            + 0.5 * l_logmag
            + 0.5 * l_sc
            + 0.01 * e_l1
        )

        return loss

    model.compile(
        optimizer=opt,
        loss=masked_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# -------------------------------- Main --------------------------------------- #

def _str2bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "t", "yes", "y"): return True
    if s in ("0", "false", "f", "no", "n"): return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True, type=Path)
    ap.add_argument("--test-manifest",  required=True, type=Path)
    ap.add_argument("--train-stats",    required=True, type=Path, help="train_feature_stats.npz")
    ap.add_argument("--outdir",         required=True, type=Path)
    ap.add_argument("--batch-size",     type=int, default=16)
    ap.add_argument("--epochs",         type=int, default=100)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--seed",           type=int, default=41)
    ap.add_argument("--mel-ceps",       type=int, default=0)

    # STRICT overrides for FE-missing params (NO defaults; raise if missing)
    ap.add_argument("--stft-center",    type=_str2bool, default=None,
                    help="REQUIRED if FeatureConfig lacks 'center' (true/false)")
    ap.add_argument("--log-base",       type=str, choices=["ln", "log10"], default=None,
                    help="REQUIRED if FeatureConfig lacks 'log_base'")
    ap.add_argument("--log-eps",        type=float, default=None,
                    help="REQUIRED if FeatureConfig lacks 'log_eps'")
    ap.add_argument("--peak-target",    type=float, default=None,
                    help="REQUIRED if FeatureConfig lacks 'peak_target'")

    # Reconstruction knobs
    ap.add_argument("--cap-ratio",      type=float, default=6.0)
    ap.add_argument("--ridge",          type=float, default=1e-3)
    ap.add_argument("--noisy-blend",    type=float, default=0.20)

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)
    (args.outdir / "wavs").mkdir(parents=True, exist_ok=True)

    # ---- Front-end ----
    fcfg = FeatureConfig(mel_ceps_keep=args.mel_ceps)
    for k in ("sr", "win_length", "hop_length", "n_fft", "fmin", "fmax"):
        if not hasattr(fcfg, k):
            raise AttributeError(f"FeatureConfig missing required attribute '{k}'")
    fx = FeatureExtractor(fcfg)

    # Pull strict parameters either from FE or CLI (NO silent defaults)
    def require_param(name: str, cli_val, fe_obj, fe_attr: str):
        if hasattr(fe_obj, fe_attr):
            return getattr(fe_obj, fe_attr)
        if cli_val is None:
            raise AttributeError(
                f"FeatureConfig missing '{fe_attr}'. Provide '--{name.replace('_','-')}' on CLI."
            )
        return cli_val

    center      = bool(require_param("stft_center", args.stft_center, fcfg, "center"))
    log_base    = str(require_param("log_base",    args.log_base,    fcfg, "log_base"))
    log_eps     = float(require_param("log_eps",   args.log_eps,     fcfg, "log_eps"))
    peak_target = float(require_param("peak_target", args.peak_target, fcfg, "peak_target"))

    # ---- Normalization scaler ----
    mel_mean, mel_std, f0_mean, f0_std = load_feature_stats(args.train_stats)

    # ---- Data ----
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

    # ---- Model ----
    sample_npz = np.load(_norm_path(tr_rows[0]["npz"]))
    D = int(sample_npz["feats"].shape[-1])
    if D < 50:
        raise ValueError(f"Model input dim must be >=50, got {D}")
    model = get_compiled_model(
        input_dim=D,
        lr=args.lr,
        mel_mean=mel_mean,
        mel_std=mel_std,
        log_base=log_base,
        log_eps=log_eps,
    )
    model.summary()

    # ---- Callbacks ----
    ckpt_path = args.outdir / "best_tf.keras"
    cbs = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss",
                        save_best_only=True, save_weights_only=False, verbose=1),
        CSVLogger(str(args.outdir / "history.csv")),
    ]

    # ---- Train ----
    try:
        print("[INFO] starting fit", flush=True)
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=args.epochs,
            callbacks=cbs,
            verbose=1,
        )
        print("[INFO] fit complete", flush=True)
    except Exception as e:
        print(f"[ERROR] fit failed: {e}", file=sys.stderr, flush=True)
        raise

    # ---- Save history JSON ----
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(args.outdir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # ------------------------ Evaluate & dump wavs (STRICT) ------------------- #
    def build_normalized_input_from_npz(npz_path: str) -> np.ndarray:
        d = np.load(npz_path, allow_pickle=False)
        feats = d["feats"].astype(np.float32)  # [T,D]
        if feats.shape[1] < 50:
            raise ValueError("Expected at least 50 feature dims.")
        mel_in = feats[:, :48]
        f0_in  = feats[:, 48:49]
        vprob  = feats[:, 49:50]
        ceps   = feats[:, 50:] if feats.shape[1] > 50 else None
        mel_in = z_norm_mel(mel_in, mel_mean, mel_std)
        f0_in  = z_norm_f0(f0_in.squeeze(-1), f0_mean, f0_std)[:, None]
        return np.concatenate([mel_in, f0_in, vprob], axis=-1) if ceps is None \
               else np.concatenate([mel_in, f0_in, vprob, ceps], axis=-1)

    metrics: Dict[str, Any] = {}
    test_rows = [df_test.iloc[i].to_dict() for i in range(min(12, len(df_test)))]

    sr         = int(fcfg.sr)
    stft_win   = int(fcfg.win_length)
    stft_hop   = int(fcfg.hop_length)
    n_fft_eff  = int(max(int(fcfg.n_fft), int(fcfg.win_length)))
    fmin       = float(fcfg.fmin)
    fmax       = fcfg.fmax  # allow None → Nyquist

    # EXACT Mel filterbank like FE
    M = 48
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft_eff,
        n_mels=M,
        fmin=fmin,
        fmax=fmax,
        htk=True
    ).astype(np.float32)  # [M, F]

    pesq_list, stoi_list, sisdr_list = [], [], []
    snr_in_list, snr_out_list = [], []
    dnsmos_sig, dnsmos_bak, dnsmos_ovr = [], [], []

    for j, row in enumerate(tqdm(test_rows, desc="Render test samples", ncols=100)):
        npz_path   = _norm_path(row["npz"])
        noisy_path = _norm_path(str(row["noisy"]))
        clean_path = _norm_path(str(row["clean"]))

        feats_n = build_normalized_input_from_npz(npz_path)  # [T,D] normalized
        X = feats_n[None, ...]
        Mmask = np.ones((1, feats_n.shape[0], 1), dtype=np.float32)

        pred_logmel_n = model.predict([X, Mmask], verbose=0)[0]         # [T̂,48] normalized
        pred_logmel   = z_denorm_mel(pred_logmel_n, mel_mean, mel_std)  # [T̂,48] true log-Mel

        noisy_wav, sr_n = sf.read(noisy_path, dtype="float32", always_2d=False)
        clean_wav, sr_c = sf.read(clean_path, dtype="float32", always_2d=False)
        if noisy_wav.ndim == 2: noisy_wav = noisy_wav.mean(axis=1)
        if clean_wav.ndim == 2: clean_wav = clean_wav.mean(axis=1)
        if sr_n != sr:
            noisy_wav = librosa.resample(noisy_wav, orig_sr=sr_n, target_sr=sr, res_type="kaiser_fast")
        if sr_c != sr:
            clean_wav = librosa.resample(clean_wav, orig_sr=sr_c, target_sr=sr, res_type="kaiser_fast")

        enh_wav = reconstruct_from_logmel_strict(
            noisy_wav=noisy_wav,
            sr=sr,
            pred_logmel=pred_logmel,
            stft_win=stft_win,
            stft_hop=stft_hop,
            n_fft_eff=n_fft_eff,
            mel_fb=mel_fb,
            log_base=log_base,
            log_eps=log_eps,
            center=center,
            peak_target=peak_target,
            cap_ratio=args.cap_ratio,
            ridge=args.ridge,
            noisy_blend=args.noisy_blend,
        )

        base = args.outdir / "wavs" / f"sample_{j:02d}"
        sf.write(str(base.with_suffix(".noisy.wav")), noisy_wav, sr)
        sf.write(str(base.with_suffix(".clean.wav")), clean_wav, sr)
        sf.write(str(base.with_suffix(".enh.wav")),   enh_wav,   sr)

        # -------- STRICT SAMPLE ALIGNMENT FOR METRICS --------
        # SNR_IN on (clean, noisy)
        clean_snr, noisy_snr = _align_pair_strict(clean_wav, noisy_wav, "clean", "noisy")
        # All others on (clean, enh)
        clean_est, enh_est   = _align_pair_strict(clean_wav, enh_wav,   "clean", "enhanced")
        # ------------------------------------------------------

        # Intrusive metrics (STRICT — must be finite)
        snr_in  = float(snr_noisy(clean_snr, noisy_snr));   _assert_finite("SNR_IN", snr_in)
        snr_out = float(snr_enhanced(clean_est, enh_est));  _assert_finite("SNR_OUT", snr_out)
        sdr     = float(sisdr(clean_est, enh_est));         _assert_finite("SI_SDR", sdr)
        pesq_v  = float(pesq_score(clean_est, enh_est, sr));_assert_finite("PESQ", pesq_v)
        stoi_v  = float(stoi_score(clean_est, enh_est, sr, extended=False)); _assert_finite("STOI", stoi_v)

        snr_in_list.append(snr_in)
        snr_out_list.append(snr_out)
        sisdr_list.append(sdr)
        pesq_list.append(pesq_v)
        stoi_list.append(stoi_v)

        # Non-intrusive metric (STRICT) — uses full enhanced clip on disk
        dns = dnsmos_wav(str(base.with_suffix(".enh.wav")))
        dnsmos_sig.append(float(dns["mos_sig"]))
        dnsmos_bak.append(float(dns["mos_bak"]))
        dnsmos_ovr.append(float(dns["mos_ovr"]))

    def _mean(xs: List[float], name: str) -> float:
        if not xs:
            raise ValueError(f"No values for {name}")
        m = float(np.mean(xs))
        _assert_finite(name, m)
        return m

    metrics: Dict[str, Any] = {
        "SNR_IN":      _mean(snr_in_list, "SNR_IN"),
        "SNR_OUT":     _mean(snr_out_list, "SNR_OUT"),
        "SI_SDR":      _mean(sisdr_list, "SI_SDR"),
        "PESQ":        _mean(pesq_list, "PESQ"),
        "STOI":        _mean(stoi_list, "STOI"),
        "DNSMOS_SIG":  _mean(dnsmos_sig, "DNSMOS_SIG"),
        "DNSMOS_BAK":  _mean(dnsmos_bak, "DNSMOS_BAK"),
        "DNSMOS_OVR":  _mean(dnsmos_ovr, "DNSMOS_OVR"),
    }

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[VAL]", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
