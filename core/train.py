#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, json, random
from pathlib import Path
from typing import Optional, List, Dict, Any

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

TRAIN_CSV = "/home/vpopescu/projects/ultra-low-power-se/manifests/train.csv"
VAL_CSV   = "/home/vpopescu/projects/ultra-low-power-se/manifests/val.csv"
SAVE_DIR  = "/home/vpopescu/projects/ultra-low-power-se/artifacts/tf_manifest_only"

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

# Pentru stabilitate maximă: doar colored noise — reintroducem reverb ulterior dacă vrei
GLOBAL_AUG_CHAIN: List[Dict[str, Any]] = [
    {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 12.0}},
]

import tensorflow as tf
from tensorflow import keras

from core.models.unet1d import build_unet1d
from core.data.dataset import build_tf_dataset, fail_fast_train_manifest
from core.utils.plotting import plot_history
from core.callbacks.lr_saver import LRSaver

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def log(msg: str): print(msg, flush=True)
def set_all_seeds(seed: int = 123):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    set_all_seeds(SEED)

    save_dir = Path(SAVE_DIR)
    ensure_dir(save_dir); ensure_dir(save_dir / "ckpt"); ensure_dir(save_dir / "logs")

    # GPU strategy, fără mixed precision (float32 end-to-end)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except Exception: pass
        strategy = tf.distribute.MirroredStrategy()
        log(f"[GPU] Found {len(gpus)} GPU(s). Using MirroredStrategy (float32).")
    else:
        strategy = tf.distribute.get_strategy()
        log("[GPU] No GPU found. Running on CPU.")

    # ===== build datasets ca înainte =====
    fail_fast_train_manifest(TRAIN_CSV, log_fn=log, default_chain=GLOBAL_AUG_CHAIN)

    log("[Step] Construiesc pipeline-ul de training (manifest-only + GLOBAL_AUG_CHAIN)...")
    train_ds, n_train = build_tf_dataset(
        csv_path=TRAIN_CSV, shuffle=True, seed=SEED, mode="train",
        sample_rate=SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, batch_size=BATCH_SIZE,
        log_fn=log, default_chain=GLOBAL_AUG_CHAIN,
    )
    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    log(f"[Info] steps_per_epoch: {steps_per_epoch}")

    log("[Step] Construiesc pipeline-ul de validare (doar pre-paired din manifest)...")
    val_ds, n_val = build_tf_dataset(
        csv_path=VAL_CSV, shuffle=False, seed=SEED+1, mode="val",
        sample_rate=SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, batch_size=BATCH_SIZE,
        log_fn=log, default_chain=None,
    )
    validation_steps = max(1, n_val // BATCH_SIZE)
    log(f"[Info] validation_steps: {validation_steps}")

    # ✨ FIX: curățăm sesiunea ÎNAINTE de a intra în scope
    tf.keras.backend.clear_session()

    with strategy.scope():
        model = build_unet1d(input_len=INPUT_LEN, base_ch=BASE_CHANNELS)
        loss = keras.losses.MeanAbsoluteError()
        metrics = [keras.metrics.MeanAbsoluteError(name="mae")]

        @tf.function
        def si_snr_tf(y_true, y_pred, eps=tf.constant(1e-8, dtype=tf.float32)):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            y_true_z = y_true - tf.reduce_mean(y_true, axis=1, keepdims=True)
            y_pred_z = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
            dot = tf.reduce_sum(y_pred_z * y_true_z, axis=1, keepdims=True)
            denom = tf.reduce_sum(y_true_z * y_true_z, axis=1, keepdims=True) + eps
            s_target = dot / denom * y_true_z
            e_noise = y_pred_z - s_target
            num = tf.reduce_sum(tf.square(s_target), axis=1) + eps
            den = tf.reduce_sum(tf.square(e_noise), axis=1) + eps
            ratio = num / den
            return 10.0 * tf.math.log(ratio) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        metrics.append(si_snr_tf)

        optimizer = keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)

    model.summary(print_fn=lambda s: log(s))

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

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1,
    )

    plot_history(history, save_dir / "logs")
    log(f"[OK] Plots saved in {save_dir / 'logs'}")
    model.save(save_dir / "model.keras", include_optimizer=False)
    model.save(save_dir / "model.h5", include_optimizer=False)
    model.export(save_dir / "model_saved")
    log(f"[OK] Model(e) salvate în {save_dir}")

    hist = history.history
    summary = {
        "const": {
            "TRAIN_CSV": TRAIN_CSV, "VAL_CSV": VAL_CSV, "SAVE_DIR": str(SAVE_DIR),
            "SAMPLE_RATE": SAMPLE_RATE, "SEGMENT_SECONDS": SEGMENT_SECONDS,
            "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "BASE_CHANNELS": BASE_CHANNELS,
            "INPUT_LEN": INPUT_LEN,
        },
        "n_train_rows": n_train, "n_val_rows": n_val,
        "steps_per_epoch": steps_per_epoch, "validation_steps": validation_steps,
        "final": {k: float(hist[k][-1]) for k in hist if len(hist[k]) > 0},
    }
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(save_dir / "logs" / "history.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[OK] Summary salvat: {save_dir / 'logs' / 'history.json'}")

if __name__ == "__main__":
    main()
