from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classic_baselines.config import DSPConfig
from classic_baselines.io import ensure_dir, load_audio, read_manifest
from classic_baselines.stft import stft_analysis

DEFAULT_PCA_MODEL = Path(__file__).resolve().parent / "artifacts" / "voicebank_demand" / "pca_model.npz"

def fit_pca_model(train_csv: str | Path, out_path: str | Path, cfg: DSPConfig) -> dict:
    rows = read_manifest(train_csv)
    sum_vec = None
    sum_outer = None
    total_frames = 0

    for row in rows:
        clean_wav, _ = load_audio(row["clean"], cfg.sr)
        spec = stft_analysis(clean_wav, cfg)
        logmag = np.log(np.maximum(np.abs(spec), cfg.eps)).T.astype(np.float64)
        if logmag.size == 0:
            continue
        if sum_vec is None:
            dim = logmag.shape[1]
            sum_vec = np.zeros(dim, dtype=np.float64)
            sum_outer = np.zeros((dim, dim), dtype=np.float64)
        sum_vec += np.sum(logmag, axis=0)
        sum_outer += logmag.T @ logmag
        total_frames += logmag.shape[0]

    if sum_vec is None or sum_outer is None or total_frames <= 0:
        raise RuntimeError("Could not fit PCA: no clean training frames were collected.")

    mean = sum_vec / total_frames
    cov = sum_outer / total_frames - np.outer(mean, mean)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total_var = float(np.sum(eigvals))
    if total_var <= 0.0:
        raise RuntimeError("PCA covariance has zero variance.")
    explained = np.cumsum(eigvals) / total_var
    keep = int(np.searchsorted(explained, cfg.pca_var_threshold, side="left") + 1)
    keep = max(1, min(int(cfg.pca_max_components), keep, eigvecs.shape[1]))

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    np.savez(
        out_path,
        mean=mean.astype(np.float32),
        components=eigvecs[:, :keep].astype(np.float32),
        eigenvalues=eigvals[:keep].astype(np.float32),
        explained_variance=explained[:keep].astype(np.float32),
        total_frames=np.int64(total_frames),
        cfg=json.dumps(cfg.to_dict()),
    )
    return load_pca_model(out_path)

def load_pca_model(path: str | Path) -> dict:
    data = np.load(str(path), allow_pickle=False)
    return {
        "mean": data["mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "eigenvalues": data["eigenvalues"].astype(np.float32),
        "explained_variance": data["explained_variance"].astype(np.float32),
        "total_frames": int(data["total_frames"]),
    }

def reconstruct_logmag_frames(logmag_frames: np.ndarray, model: dict) -> np.ndarray:
    mean = model["mean"][None, :]
    components = model["components"]
    centered = np.asarray(logmag_frames, dtype=np.float32) - mean
    projected = centered @ components
    return projected @ components.T + mean

def main() -> None:
    ap = argparse.ArgumentParser("Fit PCA baseline model on VoiceBank+DEMAND clean train data")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = DSPConfig()
    model = fit_pca_model(args.train_csv, args.out, cfg)
    print("[OK] PCA model written:", args.out)
    print("[OK] Components:", model["components"].shape[1], "Frames:", model["total_frames"])

if __name__ == "__main__":
    main()
