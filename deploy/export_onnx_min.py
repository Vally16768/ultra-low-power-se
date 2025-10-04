#!/usr/bin/env python3
# deploy/export_onnx_min.py
from __future__ import annotations
import argparse, importlib, os, sys
from pathlib import Path
from typing import Any, Dict, Callable

import torch
import yaml

# adaugă root și, dacă există, src în sys.path ca să găsim pachetele locale
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "src")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def import_callable(spec: str) -> Callable[[Dict[str, Any]], torch.nn.Module]:
    """
    Acceptă:
      - 'pkg.mod:func'
      - 'pkg.mod' (și caută 'build_model' în modul)
    """
    if ":" in spec:
        mod, fn = spec.split(":", 1)
        m = importlib.import_module(mod)
        if not hasattr(m, fn):
            raise SystemExit(f"{mod} nu are '{fn}'")
        return getattr(m, fn)
    m = importlib.import_module(spec)
    if not hasattr(m, "build_model"):
        raise SystemExit(f"{spec} nu are 'build_model'")
    return m.build_model


def load_cfg(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, help="ex: se_models.mamba_unet.model:build_model (suprascrie cfg.model.module)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="ex: artifacts/export/mamba_unet.onnx")
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--sample_len", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # sursa adevărului pentru builder
    model_spec = args.model or (cfg.get("model", {}) or {}).get("module", "se_models.mamba_unet.model")
    build = import_callable(model_spec)

    # opset / sample_len pot veni din CLI sau cfg.export
    exp_cfg = cfg.get("export", {}) or {}
    opset = int(args.opset if args.opset is not None else exp_cfg.get("opset", 17))
    T = int(args.sample_len if args.sample_len is not None else exp_cfg.get("sample_len", 16000))
    dyn = bool(exp_cfg.get("dynamic_axes", True))

    model = build(cfg)
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 1, T, dtype=torch.float32)
    dynamic_axes = {"noisy": {0: "B", 2: "T"}, "enhanced": {0: "B", 2: "T"}} if dyn else None

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            f=str(out),
            input_names=["noisy"],
            output_names=["enhanced"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"[export] scris ONNX -> {out} (opset={opset}, T={T}, dynamic={bool(dynamic_axes)})")


if __name__ == "__main__":
    main()
