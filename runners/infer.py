from __future__ import annotations
import os, csv, importlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf
import torch

from models.factory import build_model_from_cfg

# runners/infer.py
def _postproc_wave(y: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    # 1) DC remove
    y = y - np.mean(y)
    # 2) optional RMS match la input (sigur pe identitate/dummy)
    if ref is not None:
        eps = 1e-8
        rms_ref = np.sqrt(np.mean(ref**2) + eps)
        rms_y   = np.sqrt(np.mean(y**2)   + eps)
        if rms_y > 0:
            y *= (rms_ref / rms_y)
    # 3) hard-limit la [-1,1] să evităm clipping pe scriere WAV
    y = np.clip(y, -1.0, 1.0)
    return y

def _load_ckpt(ckpt_path: str | Path, cfg: Dict[str, Any], device: torch.device):
    """
    Încărcă modelul din checkpoint dacă există; altfel construiește din config.
    Acceptă ckpt-uri cu chei 'state_dict' sau direct state_dict.
    """
    ckpt_path = str(ckpt_path)
    model = build_model_from_cfg(cfg).to(device)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[infer] WARNING: checkpoint absent at {ckpt_path}. Using uninitialized model.")
        model.eval()
        return model, None

    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = blob if isinstance(blob, dict) else None
    model_cfg = cfg
    try:
        name = cfg.get("model", {}).get("name", "mamba_unet")
        mod = importlib.import_module(f"se_models.{name}")
        model = mod.build_model(cfg).to(device)
    except Exception:
        model = build_model_from_cfg(cfg).to(device)

    if state is not None:
        model.load_state_dict(state, strict=False)

    # detectează formate diferite
    if state is None and all(k.startswith("module.") for k in blob.keys()):
        state = blob
    if state is None and "model" in blob:
        state = blob["model"]

    # dacă ckpt conține config specific, permite override
    model_cfg = blob.get("config", cfg)
    try:
        # dacă există o implementare specifică în se_models.<name>
        name = model_cfg.get("model", {}).get("name", "mamba_unet")
        mod = importlib.import_module(f"se_models.{name}")
        model = mod.build_model(model_cfg).to(device)
    except Exception:
        # rămâne modelul din factory
        pass

    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[infer] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    print(f"[infer] loaded ckpt: {ckpt_path}")
    return model, model_cfg


def _write_wav(path: str | Path, x: np.ndarray, sr: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr)


def _as_tensor_1c(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convertește numpy [T] / [B,T] în tensor [B,1,T] pentru Conv1d.
    """
    t = torch.from_numpy(x).float().to(device)
    if t.dim() == 1:       # [T]
        t = t.unsqueeze(0).unsqueeze(0)   # [1,1,T]
    elif t.dim() == 2:     # [B,T]
        t = t.unsqueeze(1)                # [B,1,T]
    elif t.dim() == 3:     # [B,C,T] - lăsăm așa
        pass
    else:
        raise RuntimeError(f"Unexpected input dim {t.dim()} for audio tensor")
    return t


def _to_waveform(y: torch.Tensor) -> np.ndarray:
    """
    Acceptă [B,1,T] sau [B,T] și returnează [T] pe CPU.
    """
    if y.dim() == 3:   # [B,1,T]
        y = y[:, 0, :]
    y = y.squeeze(0)   # [T]
    return y.detach().cpu().numpy()


def _forward_streaming(
    model: torch.nn.Module,
    x: np.ndarray,
    sr: int,
    frame: int,
    hop: int,
    device: torch.device,
    state: Optional[object] = None,
) -> np.ndarray:
    """
    Streaming frame-by-frame. Încearcă mai întâi forward stateless: y = model(xb),
    iar dacă modelul cere state: y, state = model(xb, state).
    """
    out = np.zeros_like(x, dtype=np.float32)
    i = 0
    model.eval()
    with torch.no_grad():
        while i < len(x):
            seg = x[i : i + frame]
            if len(seg) < frame:
                seg = np.pad(seg, (0, frame - len(seg))).astype(np.float32)
            xb = _as_tensor_1c(seg, device)  # [1,1,frame]
            try:
                yb = model(xb)                 # stateless
            except TypeError:
                yb, state = model(xb, state)   # stateful
            y = _to_waveform(yb)               # [T]
            L = min(frame, len(out) - i)
            out[i : i + L] = y[:L]
            i += hop
    return out


def infer_manifest(cfg: Dict[str, Any], manifest_csv: str, out_dir: str, streaming: bool = False) -> str:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("infer", {}).get("use_gpu", True) else "cpu"
    )
    sr = int(cfg.get("data", {}).get("sample_rate", 16000))
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint", {}).get("dir", "artifacts/checkpoints/exp"))
    ckpt_path = ckpt_dir / "model.ckpt"

    model, _ = _load_ckpt(ckpt_path, cfg, device)
    model.to(device).eval()

    frame = int(sr * cfg.get("inference", {}).get("frame_ms", 20) / 1000)
    hop = int(sr * cfg.get("inference", {}).get("hop_ms", 10) / 1000)

    out_dir = Path(out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "pairs_eval.csv"

    rows_out: List[Dict[str, str]] = []
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            noisy_p, clean_p = r["noisy"], r["clean"]
            x, s = sf.read(noisy_p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(-1)
            if s != sr:
                try:
                    import resampy
                except Exception as e:
                    raise RuntimeError(
                        f"Sample rate mismatch: file has {s} Hz but config requires {sr} Hz. "
                        f"Install 'resampy' or pre-resample your data."
                    ) from e
                x = resampy.resample(x, s, sr).astype(np.float32)

            if streaming:
                y = _forward_streaming(model, x, sr, frame, hop, device)
            else:
                with torch.no_grad():
                    yb = model(_as_tensor_1c(x, device))
                y = _to_waveform(yb)
                y = _postproc_wave(y, ref=x)

                # streaming: după obținerea lui y:
                y = _postproc_wave(y, ref=x)

            enh_p = out_dir / "wav" / (Path(noisy_p).stem + "_enh.wav")
            _write_wav(enh_p, y, sr)
            rows_out.append({"noisy": noisy_p, "clean": clean_p, "enhanced": str(enh_p)})

    with open(out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["noisy", "clean", "enhanced"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"[infer] wrote manifest: {out_manifest}")
    return str(out_manifest)


def main(cfg: Dict[str, Any]):
    # a) offline pe test standard
    man_std = cfg.get("data", {}).get("test_std_manifest")
    if man_std:
        infer_manifest(cfg, man_std, out_dir="artifacts/infer/test_standard", streaming=False)

    # b) streaming pe challenge (dacă e activ în config și există manifest)
    streaming_enabled = bool(cfg.get("inference", {}).get("streaming", True))
    man_ch = cfg.get("data", {}).get("test_challenge_manifests", {}).get("unseen_noises")
    if streaming_enabled and man_ch:
        infer_manifest(cfg, man_ch, out_dir="artifacts/infer/challenge_streaming", streaming=True)
