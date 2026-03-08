from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_manifest(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {csv_path}")
    if not {"noisy", "clean"}.issubset(rows[0].keys()):
        raise ValueError(f"Manifest must contain 'noisy' and 'clean': {csv_path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        noisy = str(Path(row["noisy"]).expanduser())
        clean = str(Path(row["clean"]).expanduser())
        out.append({
            "noisy": noisy,
            "clean": clean,
            "stem": Path(noisy).stem,
        })
    return out

def load_audio(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
    except Exception:
        wav, sr = librosa.load(str(path), sr=target_sr, mono=True)
        wav = wav.astype(np.float32)
    if int(sr) != int(target_sr):
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr), res_type="kaiser_fast")
        sr = target_sr
    return wav.astype(np.float32), int(sr)

def save_audio(path: str | Path, wav: np.ndarray, sr: int) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), int(sr))
