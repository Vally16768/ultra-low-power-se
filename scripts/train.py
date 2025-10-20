#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
from tqdm.auto import tqdm
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from .config import Defaults
from .data import TrainSequence, ValSequence
from .model import build_keras
from .eval_utils import quick_vb_snr_improvement

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="CSV with column path_clean")
    ap.add_argument("--val", required=True, help="CSV with columns path_noisy,path_clean")
    ap.add_argument("--epochs", type=int, default=Defaults.epochs)
    ap.add_argument("--steps_per_epoch", type=int, default=Defaults.steps_per_epoch)
    ap.add_argument("--val_steps", type=int, default=Defaults.val_steps)
    ap.add_argument("--sr", type=int, default=Defaults.sr)
    ap.add_argument("--n_fft", type=int, default=Defaults.n_fft)
    ap.add_argument("--hop", type=int, default=Defaults.hop)
    ap.add_argument("--ctx", type=int, default=Defaults.ctx)
    ap.add_argument("--segment_seconds", type=float, default=Defaults.segment_seconds)
    ap.add_argument("--batch_frames", type=int, default=Defaults.batch_frames)
    ap.add_argument("--lr", type=float, default=Defaults.lr)
    ap.add_argument("--dropout", type=float, default=Defaults.dropout)
    ap.add_argument("--subset_eval", type=int, default=10, help="N files for quick SNR eval (0 to disable)")
    return ap.parse_args()

def main():
    args = parse_args()
    # Fail fast: visible device & basic checks
    assert Path(args.train).is_file(), f"Missing --train file: {args.train}"
    assert Path(args.val).is_file(), f"Missing --val file: {args.val}"

    # Read CSVs with small progress
    paths_clean = []
    with open(args.train) as f:
        reader = list(csv.DictReader(f))
        for r in tqdm(reader, desc="Loading train manifest"):
            paths_clean.append(r["path_clean"])
    if not paths_clean:
        raise RuntimeError("No paths in train manifest. Expect column 'path_clean'.")

    pairs = []
    with open(args.val) as f:
        reader = list(csv.DictReader(f))
        for r in tqdm(reader, desc="Loading val manifest"):
            pairs.append((r["path_noisy"], r["path_clean"]))
    if not pairs:
        raise RuntimeError("No pairs in val manifest. Expect columns 'path_noisy,path_clean'.")

    # Sequences (show progress in preload within Sequence classes)
    tr_seq = TrainSequence(
        paths_clean,
        chain_config=None,
        sr=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        ctx=args.ctx,
        segment_seconds=args.segment_seconds,
        batch_frames=args.batch_frames,
        rng_seed=42,
    )
    va_seq = ValSequence(
        pairs,
        sr=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        ctx=args.ctx,
        batch_frames=args.batch_frames,
    )

    # Dimensions
    input_dim = tr_seq.feat_dim
    output_dim = tr_seq.F

    # KerasRegressor (sklearn API)
    reg = KerasRegressor(
        model=build_keras,
        model__input_dim=input_dim,
        model__output_dim=output_dim,
        model__lr=args.lr,
        model__dropout=args.dropout,
        epochs=args.epochs,
        verbose=1,         # Keras built-in progress bar
        batch_size=None,   # frames-based, batching handled in Sequence
        y_required=False,  # Sequence yields (X,y)
    )

    # Callbacks: EarlyStopping + best checkpoint + CSV log
    ckpt_path = "best_mask_mlp.keras"
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    # Train (Keras shows epoch progress bar + per-epoch metrics)
    reg.fit(
        tr_seq,
        validation_data=va_seq,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        callbacks=cbs,
    )

    # Quick SNR improvement on VoiceBank subset (with a progress bar)
    if args.subset_eval > 0:
        sub = pairs[: args.subset_eval]
        avg_imp = quick_vb_snr_improvement(
            sub,
            reg,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop,
            ctx=args.ctx,
            feat_dim=tr_seq.feat_dim,
        )
        print(f"Avg SNR improvement on VoiceBank (subset {len(sub)}): {avg_imp:.2f} dB")

if __name__ == "__main__":
    main()
