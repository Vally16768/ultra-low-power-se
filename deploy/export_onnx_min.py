from __future__ import annotations
import os, importlib
from typing import Any
import torch


def _resolve_builder(spec: str):
    # "se_models.lstm_baseline.model:build_model"
    mod, fn = (spec.split(":", 1) + ["build_model"])[:2]
    m = importlib.import_module(mod)
    return getattr(m, fn)


def export_onnx(cfg: Any):
    model_spec = getattr(cfg, "model_module", None) or getattr(cfg.model, "module", None)
    if not model_spec:
        raise RuntimeError("model spec not found in cfg (model.module)")
    build_fn = _resolve_builder(model_spec)
    model = build_fn(cfg).eval()

    sr = getattr(getattr(cfg, "data", None), "sample_rate", 16000) or 16000
    onnx_cfg = getattr(getattr(cfg, "export", None), "onnx", None)
    sample_len = (getattr(onnx_cfg, "sample_len", None) if onnx_cfg else None) or (sr * 2)
    x = torch.randn(1, 1, int(sample_len), dtype=torch.float32)

    exp_name = getattr(getattr(cfg, "experiment", None), "name", "model")
    out = (getattr(onnx_cfg, "out", None) if onnx_cfg else None) or os.path.join("artifacts", "export", f"{exp_name}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dynamic_axes = True if (onnx_cfg is None) else bool(getattr(onnx_cfg, "dynamic_axes", True))
    opset = int(17 if (onnx_cfg is None) else getattr(onnx_cfg, "opset", 17))

    torch.onnx.export(
        model,
        x,
        out,
        input_names=["noisy"],
        output_names=["enhanced"],
        opset_version=opset,
        dynamic_axes={"noisy": {2: "T"}, "enhanced": {2: "T"}} if dynamic_axes else None,
        do_constant_folding=True,
    )
    return out
