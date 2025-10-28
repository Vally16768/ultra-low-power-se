#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys, json, random, hashlib, os, fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# -----------------------------------------------------------------------------#
# Utils (defined early because used by path discovery)
# -----------------------------------------------------------------------------#
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def log(msg: str): print(msg, flush=True)

# -----------------------------------------------------------------------------#
# Robust project/data path resolution
# -----------------------------------------------------------------------------#
_THIS_DIR = Path(__file__).resolve().parent  # .../ultra-low-power-se/core

def _find_repo_root(start: Path) -> Path:
    """
    Discover repository root.
    Priority:
      1) ULPSE_ROOT env var
      2) Walk upward for markers: datasets/, .git, pyproject.toml, runs/
      3) If we're in .../repo/core, use parent
      4) Fallback: start
    """
    env_root = os.environ.get("ULPSE_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if root.exists():
            return root

    markers = {"datasets", ".git", "pyproject.toml", "runs"}
    cur = start
    for _ in range(10):  # up to 10 levels up
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # Common layout: repo/core -> repo
    if (start.parent / "datasets").exists() or (start.parent / ".git").exists():
        return start.parent

    return start

def _glob_train_csvs(root: Path) -> List[Path]:
    """
    Recursively find all train.csv under root/datasets.
    Returns absolute Paths. Does not follow symlinks outside datasets.
    """
    candidates: List[Path] = []
    dsets = root / "datasets"
    if not dsets.exists():
        return candidates
    # Walk manually (faster than glob for big trees & lets us rank)
    for dirpath, dirnames, filenames in os.walk(dsets):
        if "train.csv" in filenames:
            candidates.append(Path(dirpath) / "train.csv")
    return candidates

def _rank_csv_candidates(cands: List[Path]) -> List[Tuple[int, Path]]:
    """
    Rank train.csv candidates: prefer paths containing 'voicebank' and '16k'.
    Lower score is better.
    """
    ranked: List[Tuple[int, Path]] = []
    for p in cands:
        path_str = str(p).lower()
        score = 100
        # bonuses
        if "voicebank" in path_str or "vbd" in path_str:
            score -= 40
        if "demand" in path_str:
            score -= 15
        if "16k" in path_str or "16000" in path_str:
            score -= 20
        # shallower paths preferred
        depth_penalty = len(p.parts)
        score += depth_penalty // 3
        ranked.append((score, p))
    ranked.sort(key=lambda x: (x[0], len(str(x[1]))))
    return ranked

def _resolve_data_paths() -> Tuple[Path, Path, Path]:
    """
    Resolve PROJ_ROOT, DATA_DIR, TRAIN_CSV with strong defaults,
    env overrides, and auto-discovery.

    Env overrides:
      - ULPSE_ROOT        -> repo root
      - ULPSE_DATA        -> data directory that contains train.csv/test.csv
      - ULPSE_TRAIN_CSV   -> direct path to train.csv
    """
    proj_root = _find_repo_root(_THIS_DIR)

    # 1) Direct CSV override
    env_train_csv = os.environ.get("ULPSE_TRAIN_CSV")
    if env_train_csv:
        train_csv = Path(env_train_csv).resolve()
        data_dir = train_csv.parent
        return proj_root, data_dir, train_csv

    # 2) Data dir override
    env_data = os.environ.get("ULPSE_DATA")
    if env_data:
        data_dir = Path(env_data).resolve()
        train_csv = data_dir / "train.csv"
        return proj_root, data_dir, train_csv

    # 3) Default conventional path
    data_dir = proj_root / "datasets" / "voicebank-demand" / "16k"
    train_csv = data_dir / "train.csv"
    if train_csv.exists():
        return proj_root, data_dir, train_csv

    # 4) Auto-discover train.csv anywhere under datasets/
    cands = _glob_train_csvs(proj_root)
    if cands:
        ranked = _rank_csv_candidates(cands)
        best = ranked[0][1]
        best_dir = best.parent
        log("[Paths] Auto-discovered train.csv candidates (ranked):")
        for score, p in ranked[:5]:
            log(f"  score={score:3d}  {p}")
        log(f"[Paths] -> Using: {best}")
        return proj_root, best_dir, best

    # 5) Nothing found; return conventional path (will error later with hint)
    return proj_root, data_dir, train_csv

_PROJ_ROOT, _DATA_DIR, _TRAIN_CSV = _resolve_data_paths()

# Make repo root importable (for `python core/train.py`)
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

RUN_DIR = _PROJ_ROOT / "runs" / "unet1d_voicebank16k"
SAVE_DIR = RUN_DIR

def _log_resolved_paths():
    log("[Paths] Resolved repo root: " + str(_PROJ_ROOT))
    log("[Paths] DATA_DIR: " + str(_DATA_DIR))
    log("[Paths] TRAIN_CSV: " + str(_TRAIN_CSV))
    log("[Paths] TRAIN_CSV exists: " + str(_TRAIN_CSV.exists()))
    if not _TRAIN_CSV.exists():
        # Helpful hints
        alt = _THIS_DIR.parent / "datasets" / "voicebank-demand" / "16k" / "train.csv"
        log("[Paths] Conventional expected path (sibling to core): " + str(alt))
        log("[Hint] Set one of the following and re-run:")
        log("       export ULPSE_TRAIN_CSV=/abs/path/to/train.csv")
        log("       # or")
        log("       export ULPSE_DATA=/abs/path/to/folder/with/train.csv")
        log("       # or")
        log("       export ULPSE_ROOT=/abs/path/to/repo_root  # if repo not detected")

# -----------------------------------------------------------------------------#
# Constants / Hyperparams
# -----------------------------------------------------------------------------#
SAMPLE_RATE       = 16000
SEGMENT_SECONDS   = 2.0
BATCH_SIZE        = 16
EPOCHS            = 200
SEED              = 41

BASE_CHANNELS     = 64
INPUT_LEN: Optional[int] = None

PATIENCE_ES       = 7
PATIENCE_RLR      = 3
RLR_FACTOR        = 0.5
VAL_SPLIT         = 0.10

GLOBAL_AUG_CHAIN: List[Dict[str, Any]] = [
    {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 12.0}},
]

# -----------------------------------------------------------------------------#
# Imports that rely on repo code
# -----------------------------------------------------------------------------#
import tensorflow as tf
from tensorflow import keras

from .models.unet1d import build_unet1d
from .data.dataset import build_tf_dataset, fail_fast_train_manifest
from .utils.plotting import plot_history
from .callbacks.lr_saver import LRSaver

# -----------------------------------------------------------------------------#
# More utils
# -----------------------------------------------------------------------------#
def set_all_seeds(seed: int = 123):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _stable_split_rows(rows: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    assert 0.0 < val_ratio < 1.0
    keyed = []
    for ln in rows:
        h = hashlib.md5((ln + f"|{seed}").encode("utf-8")).hexdigest()
        key = int(h[:8], 16) / 0xFFFFFFFF  # [0,1)
        keyed.append((key, ln))
    keyed.sort(key=lambda x: x[0])
    n = len(rows)
    n_val = max(1, int(round(n * val_ratio)))
    val_rows = [ln for _, ln in keyed[:n_val]]
    train_rows = [ln for _, ln in keyed[n_val:]]
    return train_rows, val_rows

def _build_train_val_csvs_from_train(train_csv: Path, out_dir: Path, val_ratio: float, seed: int) -> Tuple[Path, Path, int, int]:
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    ensure_dir(out_dir)

    with open(train_csv, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        raise RuntimeError(f"Empty CSV: {train_csv}")
    header, rows = lines[0], [ln for ln in lines[1:] if ln.strip()]

    if not rows:
        raise RuntimeError(f"No data rows found in CSV: {train_csv}")

    tr_rows, va_rows = _stable_split_rows(rows, val_ratio, seed)

    out_train = out_dir / "train_split.csv"
    out_val   = out_dir / "val_split.csv"

    with open(out_train, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(tr_rows) + ("\n" if tr_rows else ""))

    with open(out_val, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(va_rows) + ("\n" if va_rows else ""))

    return out_train, out_val, len(tr_rows), len(va_rows)

# -----------------------------------------------------------------------------#
# Metric
# -----------------------------------------------------------------------------#
@tf.function
def si_snr_tf(y_true, y_pred, eps=tf.constant(1e-8, dtype=tf.float32)):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_z = y_true - tf.reduce_mean(y_true, axis=1, keepdims=True)
    y_pred_z = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    dot   = tf.reduce_sum(y_pred_z * y_true_z, axis=1, keepdims=True)
    denom = tf.reduce_sum(y_true_z * y_true_z, axis=1, keepdims=True) + eps
    s_target = dot / denom * y_true_z
    e_noise  = y_pred_z - s_target
    num = tf.reduce_sum(tf.square(s_target), axis=1) + eps
    den = tf.reduce_sum(tf.square(e_noise), axis=1) + eps
    ratio = num / den
    return 10.0 * tf.math.log(ratio) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    set_all_seeds(SEED)

    # --- Paths resolved -------------------------------------------------------
    _log_resolved_paths()

    DATA_DIR   = _DATA_DIR
    TRAIN_CSV  = _TRAIN_CSV
    TEST_CSV   = DATA_DIR / "test.csv"  # may or may not exist

    # --- Folders --------------------------------------------------------------
    save_dir = Path(SAVE_DIR)
    ensure_dir(save_dir); ensure_dir(save_dir / "ckpt"); ensure_dir(save_dir / "logs"); ensure_dir(save_dir / "splits")

    # --- Devices / Strategy (float32) ----------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        strategy = tf.distribute.MirroredStrategy()
        log(f"[GPU] Found {len(gpus)} GPU(s). Using MirroredStrategy (float32).")
    else:
        strategy = tf.distribute.get_strategy()
        log("[GPU] No GPU found. Running on CPU.")

    # --- Build deterministic 90/10 split from train CSV ----------------------
    if not TRAIN_CSV.exists():
        # Fail early with clear instructions
        raise FileNotFoundError(
            "\n[ERROR] train.csv not found.\n"
            f"        Looked at: {TRAIN_CSV}\n"
            "        Fix by setting one of:\n"
            "          export ULPSE_TRAIN_CSV=/abs/path/to/train.csv\n"
            "          export ULPSE_DATA=/abs/path/to/folder/with/train.csv\n"
            "          export ULPSE_ROOT=/abs/path/to/repo_root\n"
        )

    log(f"[Split] Creating 90/10 train/val from: {TRAIN_CSV}")
    split_train_csv, split_val_csv, n_train_rows, n_val_rows = _build_train_val_csvs_from_train(
        TRAIN_CSV, save_dir / "splits", VAL_SPLIT, SEED
    )
    log(f"[Split] -> train rows: {n_train_rows}, val rows: {n_val_rows}")

    # --- Quick integrity check (manifest-only) --------------------------------
    fail_fast_train_manifest(split_train_csv, log_fn=log, default_chain=GLOBAL_AUG_CHAIN)

    # --- Datasets -------------------------------------------------------------
    log("[Data] Building training pipeline (with GLOBAL_AUG_CHAIN)…")
    train_ds, n_train = build_tf_dataset(
        csv_path=split_train_csv, shuffle=True, seed=SEED, mode="train",
        sample_rate=SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, batch_size=BATCH_SIZE,
        log_fn=log, default_chain=GLOBAL_AUG_CHAIN,
    )
    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    log(f"[Data] steps_per_epoch: {steps_per_epoch} (rows: {n_train})")

    log("[Data] Building validation pipeline (no augmentation)…")
    val_ds, n_val = build_tf_dataset(
        csv_path=split_val_csv, shuffle=False, seed=SEED+1, mode="val",
        sample_rate=SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, batch_size=BATCH_SIZE,
        log_fn=log, default_chain=None,
    )
    validation_steps = max(1, n_val // BATCH_SIZE)
    log(f"[Data] validation_steps: {validation_steps} (rows: {n_val})")

    # --- Model ----------------------------------------------------------------
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = build_unet1d(input_len=INPUT_LEN, base_ch=BASE_CHANNELS)
        loss = keras.losses.MeanAbsoluteError()
        metrics = [keras.metrics.MeanAbsoluteError(name="mae"), si_snr_tf]
        optimizer = keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)

    model.summary(print_fn=lambda s: log(s))

    # --- Callbacks ------------------------------------------------------------
    cbs = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "ckpt" / "best.keras"),
            save_best_only=True, monitor="val_loss", mode="min", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=RLR_FACTOR, patience=PATIENCE_RLR, min_delta=1e-4, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True, verbose=1
        ),
        LRSaver(),
        keras.callbacks.CSVLogger(str(save_dir / "logs" / "history.csv"), append=False),
        keras.callbacks.TensorBoard(log_dir=str(save_dir / "tb"), write_graph=False),
    ]

    # --- Train ----------------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1,
    )

    # --- Save & Plot ----------------------------------------------------------
    plot_history(history, save_dir / "logs")
    log(f"[OK] Plots saved in {save_dir / 'logs'}")

    model.save(save_dir / "model.keras", include_optimizer=False)
    try:
        model.save(save_dir / "model.h5", include_optimizer=False)
    except Exception as e:
        log(f"[Warn] Could not save H5: {e}")

    try:
        tf.saved_model.save(model, str(save_dir / "model_saved"))
        log("[OK] SavedModel exported.")
    except Exception as e:
        log(f"[Warn] Could not export SavedModel: {e}")

    # --- History summary ------------------------------------------------------
    hist = history.history
    summary = {
        "const": {
            "TRAIN_CSV": str(_TRAIN_CSV),
            "VAL_SPLIT": VAL_SPLIT,
            "SAVE_DIR": str(SAVE_DIR),
            "SAMPLE_RATE": SAMPLE_RATE,
            "SEGMENT_SECONDS": SEGMENT_SECONDS,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "BASE_CHANNELS": BASE_CHANNELS,
            "INPUT_LEN": INPUT_LEN,
        },
        "n_train_rows_csv": n_train_rows,
        "n_val_rows_csv": n_val_rows,
        "n_train_pipeline": n_train,
        "n_val_pipeline": n_val,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "final": {k: float(hist[k][-1]) for k in hist if len(hist[k]) > 0},
    }
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(save_dir / "logs" / "history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[OK] Summary saved: {save_dir / 'logs' / 'history.json'}")

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
