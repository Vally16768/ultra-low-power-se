#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any
import torch, onnx

from se_core.modeling import build_model_from_cfg

def main(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg).to(device)

    ckpt = cfg["export"].get("checkpoint") or str(Path(cfg["train"]["outdir"])/"checkpoints"/"best.ckpt")
    sd = torch.load(ckpt, map_location="cpu")["model"]; model.load_state_dict(sd, strict=True); model.eval()

    T = int(cfg["export"].get("sample_len", 16000))
    out_path = cfg["export"].get("out", "artifacts/export/robustnet_plus.onnx")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1,1,T, device=device)
    dyn_axes = {"input": {2:"T"}, "output": {2:"T"}}
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["output"], dynamic_axes=dyn_axes, opset_version=cfg["export"].get("opset", 17), do_constant_folding=True)
    onnx.checker.check_model(out_path)
    print(f"[export] wrote {out_path}")
