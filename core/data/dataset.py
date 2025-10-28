# -*- coding: utf-8 -*-
"""
TensorFlow tf.data input pipeline for speech enhancement (VoiceBank-DEMAND 16k).

Exposes:
- build_tf_dataset(csv_path, shuffle, seed, mode, sample_rate, segment_seconds, batch_size, log_fn, default_chain)
- fail_fast_train_manifest(csv_path, log_fn, default_chain)

CSV format
----------
The loader is flexible and accepts any of these header pairs (case-insensitive):
    noisy, clean
    noisy_path, clean_path
    mixture, target

Each row must point to existing .wav files at the same sampling rate (typically 16 kHz).

Yields
------
(x, y) where:
    x: [segment_len, 1] float32 noisy input
    y: [segment_len, 1] float32 clean target
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import tensorflow as tf


# ----------------------------------------------------------------------------- #
# Small helpers
# ----------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairRow:
    idx: int
    noisy: str
    clean: str


def _normalize_header(h: str) -> str:
    return h.strip().lower().replace(" ", "_")


def _detect_columns(header: List[str]) -> Tuple[int, int]:
    """
    Returns (noisy_col_idx, clean_col_idx) for a variety of common headers.
    Raises ValueError if not found.
    """
    hmap = { _normalize_header(h): i for i, h in enumerate(header) }
    candidates = [
        ("noisy", "clean"),
        ("noisy_path", "clean_path"),
        ("mixture", "target"),
    ]
    for n, c in candidates:
        if n in hmap and c in hmap:
            return hmap[n], hmap[c]
    # fallback: try partials
    noisy_like = next((hmap[k] for k in hmap if "noisy" in k or "mixture" in k), None)
    clean_like = next((hmap[k] for k in hmap if "clean" in k or "target" in k), None)
    if noisy_like is not None and clean_like is not None:
        return noisy_like, clean_like
    raise ValueError(
        f"Could not detect noisy/clean columns. Found headers: {list(hmap.keys())}."
    )


def _read_manifest(csv_path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            raise RuntimeError(f"Empty CSV: {csv_path}")
        noisy_idx, clean_idx = _detect_columns(header)
        for i, cols in enumerate(rdr):
            if not cols or all(not c.strip() for c in cols):
                continue
            try:
                n = cols[noisy_idx].strip()
                c = cols[clean_idx].strip()
            except IndexError:
                continue
            if not n or not c:
                continue
            rows.append(PairRow(idx=len(rows), noisy=n, clean=c))
    if not rows:
        raise RuntimeError(f"No data rows parsed from {csv_path}")
    return rows


def _exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- #
# Public: fail-fast manifest check
# ----------------------------------------------------------------------------- #

def fail_fast_train_manifest(
    csv_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    default_chain: Optional[List[Dict]] = None,
) -> None:
    """
    Reads CSV and prints friendly diagnostics (no audio is loaded into memory).
    Checks: files exist; basic stats.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with pairs.
    log_fn : callable
        Logger function; defaults to print.
    default_chain : list of dict
        Optional augmentation chain configuration (logged only).
    """
    log = log_fn or (lambda s: print(s, flush=True))

    log(f"[Manifest] Reading: {csv_path}")
    rows = _read_manifest(csv_path)
    n = len(rows)
    log(f"[Manifest] Parsed rows: {n}")

    missing_noisy = [r.noisy for r in rows if not _exists(r.noisy)]
    missing_clean = [r.clean for r in rows if not _exists(r.clean)]

    if missing_noisy:
        log(f"[ERROR] Missing NOISY files: {len(missing_noisy)} (first 3 shown)")
        for p in missing_noisy[:3]:
            log(f"  - {p}")
    if missing_clean:
        log(f"[ERROR] Missing CLEAN files: {len(missing_clean)} (first 3 shown)")
        for p in missing_clean[:3]:
            log(f"  - {p}")

    if not missing_noisy and not missing_clean:
        # Quick peek
        log("[Manifest] First 3 pairs:")
        for r in rows[:3]:
            log(f"  - noisy: {r.noisy}")
            log(f"    clean: {r.clean}")

    if default_chain:
        log(f"[AugChain] Default train chain: {default_chain}")
    else:
        log("[AugChain] No default augmentation chain configured.")

    if missing_noisy or missing_clean:
        raise FileNotFoundError("Some listed audio files are missing; please fix the CSV/paths before training.")


# ----------------------------------------------------------------------------- #
# Audio I/O + preprocessing (TensorFlow graph)
# ----------------------------------------------------------------------------- #

def _tf_read_wav_mono(path: tf.Tensor, desired_sr: int) -> tf.Tensor:
    """
    Reads a WAV file and returns mono float32 samples in [-1, 1], shape [T].
    Assumes file sample rate equals desired_sr (dataset already resampled).
    """
    audio_bytes = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    # Optional (skip strict check to avoid host->device asserts):
    # tf.debugging.assert_equal(sr, tf.cast(desired_sr, tf.int32), message="Unexpected sample rate.")
    return tf.squeeze(wav, axis=-1)  # [T]


def _pad_or_crop(x: tf.Tensor, length: int) -> tf.Tensor:
    t = tf.shape(x)[0]
    def _pad():
        pad = tf.maximum(0, length - t)
        return tf.pad(x, [[0, pad]])[:length]
    def _crop_center():
        start = (t - length) // 2
        return x[start:start + length]
    return tf.cond(t < length, _pad, _crop_center)


def _random_crop(x: tf.Tensor, length: int, seed2: tf.Tensor) -> tf.Tensor:
    """Random crop with stateless seed for determinism."""
    t = tf.shape(x)[0]
    def _pad():
        pad = tf.maximum(0, length - t)
        return tf.pad(x, [[0, pad]])[:length]
    def _crop_rand():
        max_off = t - length
        # produce deterministic offset using stateless RNG
        rnd = tf.random.stateless_uniform(shape=[], seed=tf.stack([seed2, seed2 * 1664525 + 1013904223]), minval=0.0, maxval=1.0)
        off = tf.cast(tf.floor(rnd * tf.cast(max_off + 1, tf.float32)), tf.int32)
        return x[off:off + length]
    return tf.cond(t <= length, _pad, _crop_rand)


# ----------------------------------------------------------------------------- #
# Augmentations (very lightweight, train-time only)
# ----------------------------------------------------------------------------- #

def _rms(x: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(x)) + tf.constant(eps, x.dtype))


def _colored_noise(length: tf.Tensor, color: str, seed_pair: tf.Tensor) -> tf.Tensor:
    """
    Generate colored noise using simple spectral shaping:
        white: alpha=0
        pink:  alpha=1
        brown: alpha=2
    """
    color = color.lower()
    alpha = 0.0
    if color.startswith("p"):   # pink
        alpha = 1.0
    elif color.startswith("b"): # brownian / brown
        alpha = 2.0
    elif color.startswith("w"):
        alpha = 0.0

    # start with white noise
    w = tf.random.stateless_normal([length], seed=seed_pair, dtype=tf.float32)

    if alpha == 0.0:
        return w

    # FFT shape: use rfft for efficiency
    # guard for very short signals
    n = tf.shape(w)[0]
    # next power-of-two can be stable but we'll use exact n to keep crop precise
    W = tf.signal.rfft(w)  # complex64, len n//2+1
    freqs = tf.cast(tf.range(tf.shape(W)[0]), tf.float32)  # 0..N/2
    # Avoid div by zero: f=0 bin -> no shaping (gain 1)
    # magnitude shaping ~ 1 / f^(alpha/2)
    gain = tf.where(freqs > 0.0, tf.pow(freqs, -alpha * 0.5), tf.ones_like(freqs))
    gain = tf.cast(gain, tf.complex64)
    Y = W * gain
    y = tf.signal.irfft(Y, [n])  # back to time, real64->float32
    # normalize RMS to ~1
    y = tf.cast(y, tf.float32)
    y = y / (_rms(y) + 1e-12)
    return y


def _add_colored_noise(
    x: tf.Tensor,
    snr_db: float,
    color: str,
    seed_base: tf.Tensor,
) -> tf.Tensor:
    """
    Add colored noise to achieve target SNR (dB) wrt signal x.

    Args
    ----
    x : [T] float32
    snr_db : float
    color : str ("white", "pink", "brown")
    seed_base : scalar int32/64 used for determinism
    """
    length = tf.shape(x)[0]
    seed_hi = seed_base * 1103515245 + 12345
    n = _colored_noise(length, color=color, seed_pair=tf.stack([seed_base, seed_hi]))
    # scale noise to target SNR
    rms_x = _rms(x)
    rms_n = _rms(n)
    snr_lin = tf.pow(tf.constant(10.0, tf.float32), snr_db / 20.0)  # amplitude ratio
    scale = tf.where(rms_n > 0, rms_x / (snr_lin * rms_n), tf.constant(0.0, tf.float32))
    y = x + n * scale
    return y


def _apply_aug_chain(
    x_noisy: tf.Tensor,
    chain: Optional[List[Dict]],
    seed_base: tf.Tensor,
) -> tf.Tensor:
    """Apply a minimal, whitelisted augmentation chain to the noisy input."""
    if not chain:
        return x_noisy

    y = x_noisy
    for i, step in enumerate(chain):
        name = (step.get("name") or "").lower()
        params = step.get("params") or {}
        if name == "add_colored_noise":
            color = str(params.get("color", "pink"))
            snr_db = float(params.get("snr_db", 12.0))
            y = _add_colored_noise(y, snr_db=snr_db, color=color, seed_base=seed_base + i + 1)
        else:
            # ignore any unsupported/forbidden transforms
            continue
    return y


# ----------------------------------------------------------------------------- #
# Dataset builder
# ----------------------------------------------------------------------------- #

def build_tf_dataset(
    csv_path: Path,
    shuffle: bool,
    seed: int,
    mode: str,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    log_fn: Optional[Callable[[str], None]] = None,
    default_chain: Optional[List[Dict]] = None,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a tf.data pipeline.

    Parameters
    ----------
    mode : "train" | "val"
        In "train": random crops + augmentations (only if chain provided).
        In "val"  : center crop/pad, no augmentation.
    Returns
    -------
    dataset, n_rows
    """
    assert mode in ("train", "val"), f"Unsupported mode: {mode}"
    log = log_fn or (lambda s: None)

    rows = _read_manifest(csv_path)
    n_rows = len(rows)

    # Prepare Python-side list for from_tensor_slices
    noisy_list = [r.noisy for r in rows]
    clean_list = [r.clean for r in rows]
    idx_list = [r.idx for r in rows]

    ds = tf.data.Dataset.from_tensor_slices({
        "noisy": tf.convert_to_tensor(noisy_list, dtype=tf.string),
        "clean": tf.convert_to_tensor(clean_list, dtype=tf.string),
        "idx":   tf.convert_to_tensor(idx_list,   dtype=tf.int32),
    })

    if shuffle:
        ds = ds.shuffle(buffer_size=max(1024, n_rows), seed=seed, reshuffle_each_iteration=True)

    seg_len = int(round(float(segment_seconds) * float(sample_rate)))
    seg_len = max(1, seg_len)

    # map: read files
    def _load_map(ex):
        noisy = _tf_read_wav_mono(ex["noisy"], desired_sr=sample_rate)  # [Tn]
        clean = _tf_read_wav_mono(ex["clean"], desired_sr=sample_rate)  # [Tc]
        return noisy, clean, ex["idx"]

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(_load_map, num_parallel_calls=AUTOTUNE)

    # crop/pad
    def _segment_map(noisy, clean, idx):
        # derive deterministic per-example seeds from (global seed, idx)
        # keep it graph-friendly
        seed_base = tf.cast(seed, tf.int32) * 1000003 + tf.cast(idx, tf.int32)

        if mode == "train":
            noisy_seg = _random_crop(noisy, seg_len, seed2=seed_base)
            clean_seg = _random_crop(clean, seg_len, seed2=seed_base * 3 + 7)
        else:
            noisy_seg = _pad_or_crop(noisy, seg_len)
            clean_seg = _pad_or_crop(clean, seg_len)

        # augmentations (train only)
        if mode == "train" and default_chain:
            noisy_seg = _apply_aug_chain(noisy_seg, default_chain, seed_base=seed_base)

        # add channel dim: [T, 1]
        noisy_seg = tf.expand_dims(noisy_seg, axis=-1)
        clean_seg = tf.expand_dims(clean_seg, axis=-1)
        return noisy_seg, clean_seg

    ds = ds.map(_segment_map, num_parallel_calls=AUTOTUNE)

    # batching + prefetch
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    # Log shapes for sanity
    try:
        spec = ds.element_spec
        x_spec, y_spec = spec
        log(f"[Data] element_spec X: {x_spec} | Y: {y_spec}")
        log(f"[Data] rows (manifest): {n_rows}; segment_len: {seg_len}; batch_size: {batch_size}")
    except Exception:
        pass

    return ds, n_rows
