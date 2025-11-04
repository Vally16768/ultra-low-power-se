#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_training_data.py — Build cached features for Track 1

Inputs:
  - CSV with headers: (noisy, clean) OR (noisy_path, clean_path) OR (mixture, target)
    Paths may be absolute or relative; quoted entries are supported.

Outputs (under --outdir):
  - <subset>/0000000.npz, ...  (per-utterance feature packs)
  - <subset>_manifest.parquet  (if pyarrow fastparquet installed; else only CSV)
  - <subset>_manifest.csv
  - <subset>_feature_stats.npz (mel_mean, mel_std, f0_mean, f0_std)
  - <subset>_failures.jsonl    (only if any rows failed)

Each NPZ contains (depending on config):
  - feats        [T, D] float32  (stacked: 48 logMel + F0 + vprob [+ cepstra])
  - mel_log      [T, 48] float32
  - f0_hz        [T]     float32
  - vprob        [T]     float32
  - mel_ceps     [T, K]  float32 (if --mel-ceps > 0)
  - meta         dict (sr, frame_hop, frame_len, etc.)
  - noisy_path   str
  - clean_path   str

Global stats:
  - mel_mean, mel_std: per-band over ALL frames (T) of all files
  - f0_mean, f0_std: over voiced frames only (f0 > 0)

Usage:
  python core/features/create_training_data.py \
    --csv datasets/datasets/voicebank-demand/16k/train.csv \
    --outdir runs/feat_cache \
    --subset-name train \
    --mel-ceps 0 \
    --jobs 8
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Local imports
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent.parent  # project root hint
import sys
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from core.features.feature import FeatureConfig, FeatureExtractor


ACCEPT_HEADERS = [
    ("noisy", "clean"),
    ("noisy_path", "clean_path"),
    ("mixture", "target"),
]

# ----------------------------- IO helpers ------------------------------------ #

def read_pairs(csv_path: Path) -> List[Tuple[int, str, str]]:
    """Return list of (idx, noisy_path, clean_path)."""
    rows: List[Tuple[int, str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty CSV or missing header.")
        headers = tuple(h.strip().lower() for h in reader.fieldnames)
        match = None
        for cand in ACCEPT_HEADERS:
            if all(h in headers for h in cand):
                match = cand
                break
        if match is None:
            raise ValueError(f"Unsupported CSV headers {headers}. "
                             f"Allowed: {ACCEPT_HEADERS}")

        for i, row in enumerate(reader):
            noisy = row[match[0]].strip()
            clean = row[match[1]].strip()
            # Strip wrapping quotes if any (csv already unquotes, this is extra safety)
            noisy = noisy.strip('"')
            clean = clean.strip('"')
            rows.append((i, noisy, clean))
    return rows


def ensure_abs(path_str: str, base: Optional[Path]) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve() if base else p.resolve())


# ----------------------------- Worker ---------------------------------------- #

@dataclass(frozen=True)
class WorkerConfig:
    out_dir: str
    mel_ceps_keep: int
    return_mag: bool


def process_one(i: int, noisy_path: str, clean_path: str, fx_cfg: FeatureConfig,
                wcfg: WorkerConfig) -> Dict[str, Any]:
    """
    Extract features from noisy_path and save NPZ.
    Returns a dict with manifest info and partial stats.
    """
    out_npz = Path(wcfg.out_dir) / f"{i:07d}.npz"

    # Build extractor per-process (librosa is fine with forked procs)
    fx = FeatureExtractor(fx_cfg)

    # Extract
    pack = fx.from_file(noisy_path)
    # Save NPZ
    np.savez_compressed(
        out_npz,
        feats=pack["feats"].astype(np.float32),
        mel_log=pack["mel_log"].astype(np.float32),
        f0_hz=pack["f0_hz"].astype(np.float32),
        vprob=pack["vprob"].astype(np.float32),
        mel_ceps=(pack["mel_ceps"].astype(np.float32) if pack["mel_ceps"] is not None else np.array([], dtype=np.float32)),
        meta=json.dumps(pack["meta"]),
        noisy_path=str(noisy_path),
        clean_path=str(clean_path),
    )

    # Stats contributions
    mel = pack["mel_log"]  # [T, n_mels]
    f0  = pack["f0_hz"]    # [T]
    T   = int(mel.shape[0])

    mel_sum = mel.sum(axis=0, dtype=np.float64)
    mel_sq  = (mel**2).sum(axis=0, dtype=np.float64)

    vmask = f0 > 0.0
    f0_sum = float(f0[vmask].sum(dtype=np.float64)) if np.any(vmask) else 0.0
    f0_sq  = float((f0[vmask]**2).sum(dtype=np.float64)) if np.any(vmask) else 0.0
    f0_cnt = int(vmask.sum())

    return {
        "idx": i,
        "npz": str(out_npz),
        "noisy": str(noisy_path),
        "clean": str(clean_path),
        "n_frames": T,
        "mel_sum": mel_sum,
        "mel_sq": mel_sq,
        "f0_sum": f0_sum,
        "f0_sq": f0_sq,
        "f0_cnt": f0_cnt,
    }


# ----------------------------- Main ------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="Input pairs CSV (noisy,clean)")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory (will be created)")
    ap.add_argument("--subset-name", default="train", choices=["train", "val", "test"])
    ap.add_argument("--root", type=Path, default=None, help="Optional root to prepend for relative paths")
    ap.add_argument("--mel-ceps", type=int, default=0, help="If >0, keep K cepstral coeffs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    ap.add_argument("--return-mag", action="store_true", help="Also compute STFT magnitude (debug only; increases NPZ size)")
    args = ap.parse_args()

    # Resolve output directories
    args.outdir.mkdir(parents=True, exist_ok=True)
    feats_dir = args.outdir / args.subset_name
    feats_dir.mkdir(parents=True, exist_ok=True)

    # Read and absolutize pairs
    pairs = read_pairs(args.csv)
    base = args.root if args.root else args.csv.parent
    pairs_abs = [(i, ensure_abs(n, base), ensure_abs(c, base)) for (i, n, c) in pairs]

    # Feature extractor config
    fx_cfg = FeatureConfig(
        # Keep defaults for causal front-end
        n_fft=320,               # auto-upgraded to >= win_length internally
        mel_ceps_keep=args.mel_ceps,
        return_mag=args.return_mag,
    )

    # Worker config
    wcfg = WorkerConfig(
        out_dir=str(feats_dir),
        mel_ceps_keep=args.mel_ceps,
        return_mag=args.return_mag,
    )

    print(f"[CFG] subset={args.subset_name}  out={feats_dir}")
    print(f"[CFG] rows={len(pairs_abs)}  jobs={args.jobs}  mel_ceps={args.mel_ceps}")

    # Parallel extraction
    manifest_rows: List[Dict[str, Any]] = []
    mel_running_sum: Optional[np.ndarray] = None
    mel_running_sq: Optional[np.ndarray] = None
    f0_sum_total = 0.0
    f0_sq_total  = 0.0
    f0_count     = 0

    failures_path = args.outdir / f"{args.subset_name}_failures.jsonl"
    has_failures = False

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = []
        for i, noisy, clean in pairs_abs:
            futs.append(ex.submit(process_one, i, noisy, clean, fx_cfg, wcfg))

        for fut in as_completed(futs):
            try:
                res = fut.result()
                manifest_rows.append({
                    "idx": res["idx"],
                    "npz": res["npz"],
                    "noisy": res["noisy"],
                    "clean": res["clean"],
                    "n_frames": res["n_frames"],
                })
                ms, msq = np.asarray(res["mel_sum"], dtype=np.float64), np.asarray(res["mel_sq"], dtype=np.float64)
                if mel_running_sum is None:
                    mel_running_sum = ms
                    mel_running_sq  = msq
                else:
                    mel_running_sum += ms
                    mel_running_sq  += msq

                f0_sum_total += float(res["f0_sum"])
                f0_sq_total  += float(res["f0_sq"])
                f0_count     += int(res["f0_cnt"])

            except Exception as e:
                # Capture failure
                has_failures = True
                err = {"error": str(e)}
                try:
                    # Try to attach minimal info if available
                    args_tuple = fut._args if hasattr(fut, "_args") else None  # not public API, best-effort
                except Exception:
                    args_tuple = None
                rec = {"args": str(args_tuple), "error": str(e)}
                with open(failures_path, "a") as ff:
                    ff.write(json.dumps(rec) + "\n")
                print(f"[WARN] worker failed: {e}")

    # Sort manifest by idx for determinism
    manifest_rows.sort(key=lambda r: r["idx"])

    # Save manifest (CSV + Parquet if possible)
    import pandas as pd
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = args.outdir / f"{args.subset_name}_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[OK] manifest CSV: {manifest_csv}")

    try:
        manifest_parquet = args.outdir / f"{args.subset_name}_manifest.parquet"
        manifest_df.to_parquet(manifest_parquet, index=False)  # requires pyarrow or fastparquet
        print(f"[OK] manifest Parquet: {manifest_parquet}")
    except Exception as e:
        print(f"[NOTE] Could not write parquet ({e}). CSV is available.")

    # Compute & save global stats
    n_frames_total = int(sum(r["n_frames"] for r in manifest_rows))
    if mel_running_sum is None or mel_running_sq is None or n_frames_total == 0:
        raise RuntimeError("No features were produced; cannot compute stats.")

    mel_mean = (mel_running_sum / max(1, n_frames_total)).astype(np.float32)
    mel_var  = (mel_running_sq / max(1, n_frames_total) - mel_mean**2).astype(np.float32)
    mel_std  = np.sqrt(np.maximum(mel_var, 1e-8)).astype(np.float32)

    if f0_count > 0:
        f0_mean = np.float32(f0_sum_total / f0_count)
        f0_var  = np.float32(f0_sq_total  / f0_count - float(f0_mean)**2)
        f0_std  = np.float32(np.sqrt(max(f0_var, 1e-8)))
    else:
        f0_mean, f0_std = np.float32(0.0), np.float32(1.0)

    stats_npz = args.outdir / f"{args.subset_name}_feature_stats.npz"
    np.savez(stats_npz, mel_mean=mel_mean, mel_std=mel_std,
             f0_mean=f0_mean, f0_std=f0_std)
    print(f"[OK] stats: {stats_npz}")
    if has_failures:
        print(f"[WARN] Some items failed. See: {failures_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
