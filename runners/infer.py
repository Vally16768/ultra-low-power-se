from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import torch, soundfile as sf, numpy as np
import torch.nn as nn

def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_name = cfg.get("model",{}).get("name","mamba_unet")
    module = __import__(f"se_models.{model_name}.model", fromlist=["*"])
    return module.build_model(cfg)

def main(cfg: Dict[str, Any]):
    in_wav = cfg.get("inference",{}).get("in_wav")
    out_wav = cfg.get("inference",{}).get("out_wav","artifacts/demo_out.wav")
    sr = int(cfg.get("data",{}).get("sample_rate",16000))

    if in_wav and Path(in_wav).exists():
        x, sr0 = sf.read(in_wav)
        if x.ndim>1: x = x.mean(axis=1)
        if sr0 != sr:
            raise SystemExit(f"Sample-rate mismatch: file={sr0} vs cfg={sr}")
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        T = sr * int(cfg.get("inference",{}).get("seconds",1))
        x_t = torch.randn(1,1,T)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device).eval()
    with torch.no_grad():
        y = model(x_t.to(device)).squeeze().cpu().numpy()
    sf.write(out_wav, y, sr)
    print(f"[infer] wrote {out_wav}")