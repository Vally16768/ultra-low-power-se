#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifică un model ONNX:
  - onnx.checker.check_model
  - infer_shapes
  - onnxsim.simplify (opțional)
  - onnxruntime smoke-run pe CPU
  - paritate vs. PyTorch (dacă dai și --config)
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, hashlib, sys
import onnx
from onnx import checker, shape_inference
import numpy as np
import onnxruntime as ort
import torch
from typing import Tuple

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_sidecar(onnx_path: Path):
    side = onnx_path.with_suffix(onnx_path.suffix + ".json")
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    return {}

def _guess_in_out(sess, side: dict) -> Tuple[str, str]:
    in_name = side.get("input_name") or side.get("input") or sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    preferred = side.get("output_names") or [side.get("output")] or []
    out_name = preferred[0] if preferred and preferred[0] in out_names else out_names[0]
    return in_name, out_name

def structural_checks(onnx_path: Path) -> None:
    print(f"[check] file: {onnx_path}  size={onnx_path.stat().st_size/1e6:.2f} MB  sha256={_sha256(onnx_path)[:16]}...")
    m = onnx.load(str(onnx_path))
    checker.check_model(m)
    print("[check] onnx.checker: OK")
    _ = shape_inference.infer_shapes(m)
    print("[check] shape_inference: OK")
    try:
        from onnxsim import simplify
        ms, ok = simplify(m)
        if ok:
            tmp = onnx_path.with_suffix(".sim.onnx")
            onnx.save(ms, str(tmp))
            print(f"[check] onnxsim: OK → {tmp}")
    except Exception as e:
        print(f"[check] onnxsim skipped: {e!r}")

def smoke_runtime(onnx_path: Path, T: int | None = None):
    side = _load_sidecar(onnx_path)
    T0 = int(side.get("example_input_shape", [1,1,16000])[-1])
    if T is None: T = T0
    x = np.random.randn(1, 1, T).astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name, out_name = _guess_in_out(sess, side)
    outs = sess.run(None, {in_name: x})
    names = [o.name for o in sess.get_outputs()]
    out_map = dict(zip(names, outs))
    y = out_map.get(out_name, outs[0])

    # normalize shapes to [B,1,T]
    if y.ndim == 2:
        y = y[:, None, :]
    print(f"[run] onnxruntime: OK  in={x.shape} out={y.shape}")
    assert np.isfinite(y).all(), "Output has NaN/Inf"
    assert np.abs(y).mean() > 1e-9, "Output is (almost) zero"
    return y, side

def parity_with_torch(onnx_path: Path, cfg_path: str) -> None:
    from se_cli.config import load_config
    cfg = load_config(cfg_path, {})
    model_name = cfg.get("model", {}).get("name", "mamba_unet")
    module = __import__(f"se_models.{model_name}.model", fromlist=["build_model","Net"])
    net = module.build_model(cfg) if hasattr(module,"build_model") else module.Net(cfg)
    net.eval().cpu()

    # local wrapper → forțează [B,1,T]
    class _KeepChWrapper(torch.nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, xx):
            y = self.n(xx)
            if isinstance(y, (tuple, list)): y = y[0]
            if isinstance(y, torch.Tensor) and y.dim() == 2: y = y.unsqueeze(1)
            return y
    net = _KeepChWrapper(net).eval()

    y_o, _ = smoke_runtime(onnx_path)
    T = y_o.shape[-1]
    x_t = torch.from_numpy(np.random.randn(1,1,T).astype(np.float32))
    with torch.no_grad():
        y_t = net(x_t).cpu().numpy()

    if y_o.ndim == 2: y_o = y_o[:, None, :]
    if y_t.ndim == 2: y_t = y_t[:, None, :]

    l2 = np.linalg.norm(y_t - y_o) / (np.linalg.norm(y_t) + 1e-12)
    mae = np.mean(np.abs(y_t - y_o))
    print(f"[parity] L2-rel={l2:.3e}  MAE={mae:.3e}  (tol: L2<=1e-2 e ok pentru multe modele)")
    if l2 < 1e-2: print("[parity] OK")
    else: print("[parity] WARN: paritatea e slabă (verifică opset/dynamic_axes/padding).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="calea către modelul .onnx")
    ap.add_argument("--config", help="yaml de config (opțional, pentru paritate)")
    args = ap.parse_args()

    p = Path(args.onnx)
    if not p.exists():
        print(f"Nu există fișierul: {p}", file=sys.stderr)
        sys.exit(2)

    structural_checks(p)
    smoke_runtime(p)
    if args.config:
        parity_with_torch(p, args.config)

if __name__ == "__main__":
    main()
