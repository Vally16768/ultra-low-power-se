# runners/infer.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Iterable
from pathlib import Path
import importlib
import csv

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from ._resolve import resolve_pathlike


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _load_manifest(p: str | Path) -> Iterable[Tuple[str, str]]:
    with open(p, "r", newline="") as f:
        r = csv.DictReader(f)
        if "noisy" not in r.fieldnames or "clean" not in r.fieldnames:
            raise SystemExit(f"[infer] Manifestul {p} trebuie să aibă coloanele: noisy, clean")
        for row in r:
            yield row["noisy"], row["clean"]


def _resolve_builder(mod_path: str):
    if ":" in mod_path:
        pkg, fn = mod_path.split(":", 1)
        mod = importlib.import_module(pkg)
        return getattr(mod, fn)
    mod = importlib.import_module(mod_path)
    if hasattr(mod, "build_model"):
        return getattr(mod, "build_model")
    if hasattr(mod, "Net"):
        def _wrap(cfg=None):
            try:
                return mod.Net(cfg)
            except TypeError:
                return mod.Net()
        return _wrap
    raise SystemExit(f"[infer] Nu găsesc builder în modulul: {mod_path}")


def _strip_module(sd):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in sd.items():
        out[k[7:]] = v if k.startswith("module.") else v
    return out

def _try_load_ckpt(model, ckpt_path: str):
    import torch
    from pathlib import Path
    p = Path(ckpt_path).expanduser()
    if not p.exists():
        print(f"[infer] WARNING: checkpoint inexistent: {p}")
        return False
    obj = torch.load(str(p), map_location="cpu")
    sd = None
    for key in ("state_dict", "model", "net", "weights"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            sd = obj[key]; break
    if sd is None:
        sd = obj if isinstance(obj, dict) else None
    if not isinstance(sd, dict):
        print(f"[infer] WARNING: format checkpoint neînțeles: {type(obj)}")
        return False
    sd = _strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[infer] loaded checkpoint: {p.name}  (missing={len(missing)}, unexpected={len(unexpected)})")
    return True

def _autodiscover_ckpt(cfg):
    from pathlib import Path
    exp = cfg.get("experiment", {})
    root = Path(exp.get("out_dir", "artifacts/exp")) / exp.get("name", "")
    cands = []
    if root.exists():
        cands += list(root.rglob("*.ckpt"))
        cands += list(root.rglob("*.pt"))
        cands += list(root.rglob("*.pth"))
    if not cands:
        return None
    # preferă “best” apoi cele mai noi
    cands = sorted(cands, key=lambda p: (("best" not in p.name.lower()), -p.stat().st_mtime))
    return str(cands[0])

def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    modpath = cfg.get("model", {}).get("module", "se_models.mamba_unet.model:build_model")
    builder = _resolve_builder(modpath)
    try:
        model = builder(cfg)
    except TypeError:
        model = builder()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from pathlib import Path
    ckpt_cfg = cfg.get("model", {}).get("checkpoint")
    ckpt_eff = None
    if ckpt_cfg and Path(ckpt_cfg).expanduser().exists():
        ckpt_eff = ckpt_cfg
        print(f"[infer] using checkpoint from cfg: {ckpt_eff}")
    else:
        if ckpt_cfg:
            print(f"[infer] WARNING: checkpoint inexistent în cfg: {ckpt_cfg} → încerc autodiscover")
        auto = _autodiscover_ckpt(cfg)
        if auto:
            ckpt_eff = auto
            print(f"[infer] autodiscovered checkpoint: {ckpt_eff}")

    if ckpt_eff:
        _try_load_ckpt(model, ckpt_eff)
    else:
        print("[infer] WARNING: rulez FĂRĂ checkpoint (model random)")

    return model


@torch.no_grad()
def enhance_dataset(cfg: Dict[str, Any], manifest: str | Path, outdir: str | Path, sr: int) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for noisy_path, _clean_path in _load_manifest(manifest):
        npath = Path(noisy_path)
        x, srx = sf.read(npath, always_2d=False)
        x = _mono(np.asarray(x, dtype=np.float32))
        if srx != sr:
            raise SystemExit(f"[infer] SR diferit ({srx}) pentru {npath}. Resamplează la {sr} Hz.")

        xt = torch.from_numpy(x).float().to(device).view(1, 1, -1)
        y = model(xt)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.squeeze().detach().cpu().numpy()

        out_wav = outdir / npath.name
        sf.write(out_wav, y, sr)
    print(f"[infer] wrote enhanced wavs to {outdir}")


def main(cfg: Dict[str, Any]) -> int:
    sr = int(cfg.get("eval", {}).get("sr") or cfg.get("data", {}).get("sample_rate", 16000))

    manifest = (
        cfg.get("inference", {}).get("manifest")
        or cfg.get("eval", {}).get("offline", {}).get("manifest")
        or cfg.get("data", {}).get("manifests", {}).get("test_offline")
    )
    manifest_res = resolve_pathlike(manifest, cfg)
    outdir = resolve_pathlike(
        cfg.get("eval", {}).get("offline", {}).get("outdir") or "artifacts/eval/mamba_unet/enhanced",
        cfg
    )
    if manifest_res:
        enhance_dataset(cfg, manifest_res, outdir, sr=sr)
        return 0

    # fallback single-file demo
    in_wav = resolve_pathlike(cfg.get("inference", {}).get("in_wav"), cfg)
    out_wav = resolve_pathlike(cfg.get("inference", {}).get("out_wav", "artifacts/demo_out.wav"), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device).eval()

    if in_wav and Path(in_wav).exists():
        x, sr0 = sf.read(in_wav)
        x = _mono(np.asarray(x, dtype=np.float32))
        if sr0 != sr:
            raise SystemExit(f"Sample-rate mismatch: file={sr0} vs cfg={sr}")
        xt = torch.from_numpy(x).float().to(device).view(1, 1, -1)
    else:
        T = sr * int(cfg.get("inference", {}).get("seconds", 1))
        xt = torch.randn(1, 1, T, device=device)

    with torch.no_grad():
        y = model(xt)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.squeeze().detach().cpu().numpy()

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y, sr)
    print(f"[infer] wrote {out_wav}")
    return 0
