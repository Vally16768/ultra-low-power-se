#!/usr/bin/env python3
"""
Inferență offline pentru Ultra-Low-Power SE.

- Citește manifest CSV cu coloane: noisy[, clean]
- Încarcă modelul conform configului + checkpoint-ul (safe: weights_only=True)
- Rulează pe GPU dacă e disponibil
- Scrie fișierele enhanced *.wav în outdir

Puncte de intrare:
- enhance_dataset(cfg)  -> folosit de runners/evaluate.py
- main(cfg)             -> wrapper care face același lucru și returnează outdir
"""

from __future__ import annotations
import os
import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import soundfile as sf
import torch
from torch import nn
from tqdm import tqdm


# ---------------------- Utils ----------------------
def _resolve_build_fn(module_spec: str):
    """
    Acceptă:
      - "pkg.subpkg.module:build_model"  -> returnează funcția build_model
      - "pkg.subpkg.module"              -> caută atributul `build_model`
    """
    if ":" in module_spec:
        mod, fn = module_spec.split(":", 1)
        m = importlib.import_module(mod)
        if not hasattr(m, fn):
            raise RuntimeError(f"Nu găsesc factory '{fn}' în modulul '{mod}'")
        return getattr(m, fn)
    # fallback: caută build_model
    m = importlib.import_module(module_spec)
    if not hasattr(m, "build_model"):
        raise RuntimeError(f"Modulul '{module_spec}' nu conține 'build_model'")
    return m.build_model


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_state_safely(model: nn.Module, ckpt_path: Path, strict_load: bool = True) -> Tuple[List[str], List[str]]:
    """
    Încarcă un checkpoint PyTorch în siguranță (fără pickle arbitrar).
    - torch.load(..., weights_only=True) -> PyTorch 2.4+
    - Acceptă structuri {state_dict=..., ...} sau dict direct de greutăți.
    - Elimină prefixul "module." dacă checkpoint-ul provine din DataParallel.
    """
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model", "net")):
        state = obj.get("state_dict") or obj.get("model") or obj.get("net")
    else:
        state = obj  # probabil e direct un state_dict

    if not isinstance(state, dict):
        raise RuntimeError(f"Structură checkpoint neașteptată în: {ckpt_path}")

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)  # diagnostic mai întâi

    if missing or unexpected:
        print(f"[infer] DIAGNOSTIC mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) <= 8 and len(unexpected) <= 8:
            if missing:
                print("  missing:", missing)
            if unexpected:
                print("  unexpected:", unexpected)

    if strict_load:
        # Reîncercăm strict ca să fail-fast dacă există diferențe reale
        model.load_state_dict(state, strict=True)
        return [], []

    return list(missing), list(unexpected)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


# ---------------------- Dataclass cfg ----------------------
@dataclass
class EvalPaths:
    manifest: Path
    outdir: Path


@dataclass
class Cfg:
    sample_rate: int
    model_module: str
    checkpoint: Path
    eval_paths: EvalPaths


def _build_cfg(cfg: Dict[str, Any]) -> Cfg:
    # data.sample_rate
    sr = int(cfg.get("data", {}).get("sample_rate", 16000))

    # model.module (poate veni și din env: MODEL_MODULE)
    model_mod = cfg.get("model", {}).get("module") or os.getenv("MODEL_MODULE")
    if not model_mod:
        # fallback compat vechi
        model_mod = cfg.get("model", {}).get("MODEL_MODULE") or "se_models.mamba_unet.model:build_model"

    # inference.checkpoint (preferăm explicit)
    ckpt = cfg.get("inference", {}).get("checkpoint")
    if not ckpt:
        maybe = Path("artifacts/exp/mamba_unet_v0/best.ckpt")
        if maybe.exists():
            print(f"[infer] WARNING: checkpoint inexistent în cfg -> folosesc fallback: {maybe}")
            ckpt = str(maybe)
        else:
            raise SystemExit("Nu ai setat inference.checkpoint și nu găsesc fallback.")

    # eval.manifest + eval.outdir
    manifest = cfg.get("eval", {}).get("manifest") or cfg.get("eval", {}).get("offline_manifest") or "data/prepared/voicebank/test/manifests/pairs.csv"
    outdir = cfg.get("eval", {}).get("outdir") or "artifacts/eval/mamba_unet/enhanced"

    return Cfg(
        sample_rate=sr,
        model_module=model_mod,
        checkpoint=Path(ckpt),
        eval_paths=EvalPaths(
            manifest=Path(manifest),
            outdir=Path(outdir),
        ),
    )


# ---------------------- Infer core ----------------------
def _read_manifest(path: Path) -> List[tuple[str, str | None]]:
    rows: List[tuple[str, str | None]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or [])}
        if "noisy" not in cols:
            raise RuntimeError(f"Manifest-ul trebuie să conțină coloana 'noisy': {path}")
        for r in reader:
            noisy = r["noisy"]
            clean = r.get("clean")
            rows.append((noisy, clean))
    return rows


def _enhance_file(model: nn.Module, wav: np.ndarray, device: torch.device) -> np.ndarray:
    wav = _to_mono(wav.astype(np.float32))
    t = torch.from_numpy(wav)[None, None, :]  # [B=1, C=1, T]
    t = t.to(device, non_blocking=True)
    with torch.no_grad():
        out = model(t)
    if out.ndim == 3:  # [B,1,T]
        out = out[:, 0, :]
    out = out.squeeze(0).detach().cpu().float().numpy()
    return out


def _prepare_model(C: Cfg, cfg: Dict[str, Any]) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device: {device}")

    build_fn = _resolve_build_fn(C.model_module)
    model: nn.Module = build_fn(cfg) if build_fn.__code__.co_argcount else build_fn()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] total params: {n_params / 1e6:.2f}M")

    if not C.checkpoint.exists():
        raise SystemExit(f"Checkpoint inexistent: {C.checkpoint}")
    _load_state_safely(model, C.checkpoint, strict_load=True)

    model.to(device)
    model.eval()
    return model, device


def enhance_dataset(
    cfg: Dict[str, Any],
    manifest: str | None = None,
    outdir: str | None = None,
    sr: int | None = None,
) -> str:
    """
    Punctul de intrare folosit de runners/evaluate.py.
    Permite suprascrierea manifest/outdir/sr prin argumente.
    Returnează calea outdir unde s-au scris fișierele enhanced.
    """
    C = _build_cfg(cfg)

    # suprascrieri venite din evaluate.py
    if sr is not None:
        C.sample_rate = int(sr)
    if manifest is not None:
        C.eval_paths.manifest = Path(manifest)
    if outdir is not None:
        C.eval_paths.outdir = Path(outdir)

    C.eval_paths.outdir.mkdir(parents=True, exist_ok=True)

    model, device = _prepare_model(C, cfg)
    pairs = _read_manifest(C.eval_paths.manifest)
    print(f"[infer] {len(pairs)} fișiere din manifest: {C.eval_paths.manifest}")
    print(f"[infer] sample_rate={C.sample_rate}  outdir={C.eval_paths.outdir}")

    for noisy, _ in tqdm(pairs, desc="enhance", unit="file"):
        x, sr_file = sf.read(noisy, always_2d=False)
        if sr_file != C.sample_rate:
            raise RuntimeError(f"SR curent ({sr_file}) diferă de config ({C.sample_rate}) pentru: {noisy}")
        y = _enhance_file(model, x, device)
        sf.write(C.eval_paths.outdir / Path(noisy).name, y, samplerate=sr_file)

    print(f"[infer] wrote enhanced wavs to {C.eval_paths.outdir}")
    return str(C.eval_paths.outdir)


def main(cfg: Dict[str, Any]) -> str:
    """Wrapper compatibil cu runnerul generic; returnează outdir."""
    return enhance_dataset(cfg)


if __name__ == "__main__":
    raise SystemExit("Rulează prin: python -m se_cli.cli eval --config <yaml>")
