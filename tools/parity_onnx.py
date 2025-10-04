# tools/parity_onnx.py
from __future__ import annotations
import argparse, importlib
import numpy as np
import onnx, onnxruntime as ort
import torch
import yaml


def import_callable(spec):
    if ":" in spec:
        mod, fn = spec.split(":", 1)
        m = importlib.import_module(mod)
        return getattr(m, fn)
    m = importlib.import_module(spec)
    return m.build_model


def sisdr(x, s, eps=1e-8):
    s = s - s.mean()
    x = x - x.mean()
    a = (np.dot(x, s) / (np.dot(s, s) + eps)) * s
    e = x - a
    return 10 * np.log10((np.dot(a, a) + eps) / (np.dot(e, e) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--T", type=int, default=16000)
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    build = import_callable(args.model)
    m = build(cfg).eval()

    x = torch.randn(1, 1, args.T)
    with torch.no_grad():
        y_pt = m(x).cpu().numpy()

    onnx.checker.check_model(onnx.load(args.onnx))
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    y_ox = sess.run(["enhanced"], {"noisy": x.numpy().astype(np.float32)})[0]

    mse = np.mean((y_pt - y_ox) ** 2)
    sdr = sisdr(y_ox[0, 0], y_pt[0, 0])
    max_abs = np.max(np.abs(y_pt - y_ox))
    n_bad = int(np.sum(np.abs(y_pt - y_ox) > args.tol))
    print(f"MSE={mse:.6e}  SI-SDR(ONNX vs PT)={sdr:.2f} dB  max|Δ|={max_abs:.3e}  bad_samples>{args.tol} = {n_bad}")


if __name__ == "__main__":
    main()
