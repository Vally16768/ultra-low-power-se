#!/usr/bin/env python3
# deploy/export_onnx_min.py
import argparse, importlib, yaml, os
from pathlib import Path
import torch

def import_callable(spec: str):
    # "pkg.mod:func"
    if ":" not in spec:
        raise SystemExit(f"--model trebuie să fie 'modul:callable', primit: {spec}")
    mod, fn = spec.split(":", 1)
    m = importlib.import_module(mod)
    if not hasattr(m, fn):
        raise SystemExit(f"{mod} nu are {fn}")
    return getattr(m, fn)

def load_cfg(path: str | None):
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ex: se_models.mamba_unet.model:build_model")
    ap.add_argument("--config", required=False)
    ap.add_argument("--out", required=True, help="ex: artifacts/export/mamba_unet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--sample_len", type=int, default=16000, help="fallback T dacă nu vine din config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    build = import_callable(args.model)
    model = build(cfg)
    model.eval()

    # T fallback sau din config
    T = int(cfg.get("export", {}).get("sample_len", args.sample_len))
    dummy = torch.zeros(1, 1, T, dtype=torch.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "noisy": {0: "B", 2: "T"},
        "enhanced": {0: "B", 2: "T"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,),
            f=str(out),
            input_names=["noisy"],
            output_names=["enhanced"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"[export] scris ONNX -> {out}")

if __name__ == "__main__":
    main()
