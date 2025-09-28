#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_name = cfg.get("model",{}).get("name","mamba_unet")
    module = __import__(f"se_models.{model_name}.model", fromlist=["*"])
    return module.build_model(cfg)

def export_onnx(cfg: Dict[str, Any], opset_override: Optional[int] = None) -> Path:
    onnx_cfg = cfg.get("export", {}).get("onnx", {})
    sr = int(cfg.get("data", {}).get("sample_rate", 16000))
    frame_ms = int(cfg.get("inference", {}).get("frame_ms", 20))
    T = max(sr * frame_ms // 1000, 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg).to(device).eval()

    ckpt = cfg.get("export",{}).get("checkpoint")
    if ckpt and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    dummy = torch.randn(1, 1, T, device=device)

    out_dir = Path(cfg.get("export",{}).get("out_dir","artifacts/export"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.get("export",{}).get("name","mamba_unet_v0")
    onnx_path = out_dir / f"{name}.onnx"

    in_name  = onnx_cfg.get("input_name",  "noisy")
    out_name = onnx_cfg.get("output_name", "denoised")
    opset = int(onnx_cfg.get("opset", 18 if opset_override is None else opset_override))

    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=[in_name], output_names=[out_name],
        dynamic_axes={in_name: {2: "T"}, out_name: {1: "T"}},  # output is (B,T)
        opset_version=opset, do_constant_folding=True
    )
    print(f"[export] ONNX saved to: {onnx_path}")
    return onnx_path
