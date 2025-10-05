#!/usr/bin/env python3
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import hashlib, json

import onnx, onnxsim, onnxruntime as ort
import numpy as np


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(onnx_path: str):
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("[check] onnx.checker: OK")
    m = onnx.shape_inference.infer_shapes(m)
    print("[check] shape_inference: OK")
    return m


def _simplify(onnx_path: str) -> str:
    out = Path(onnx_path).with_suffix(".sim.onnx")
    m = onnx.load(onnx_path)
    sm, ok = onnxsim.simplify(m)
    if ok:
        onnx.save(sm, out)
        print(f"[check] onnxsim: OK → {out}")
        return str(out)
    else:
        print("[check] onnxsim: FAILED → folosesc originalul")
        return onnx_path


def _load_sidecar(onnx_path: Path):
    side = onnx_path.with_suffix(onnx_path.suffix + ".json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _guess_in_out(sess, side: dict):
    in_name = side.get("input_name") or side.get("input") or sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    preferred = side.get("output_names") or [side.get("output")] or []
    out_name = preferred[0] if preferred and preferred[0] in out_names else out_names[0]
    return in_name, out_name


def _run_once(onnx_path: str, T: int = 16000):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name, out_name = _guess_in_out(sess, _load_sidecar(Path(onnx_path)))
    x = np.random.randn(1, 1, T).astype(np.float32)
    y = sess.run([out_name], {in_name: x})[0]
    if y.ndim == 2:
        y = y[:, None, :]
    print(f"[run] onnxruntime: OK  in={(1, 1, T)} out={y.shape}")
    if not np.isfinite(y).all():
        raise SystemExit("Output conține NaN/Inf")
    return sess


def _bench(sess: ort.InferenceSession, secs: float = 5.0, T: int = 16000, sr: int = 16000):
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    x = np.random.randn(1, 1, T).astype(np.float32)
    for _ in range(5):
        sess.run([out_name], {in_name: x})
    t0 = time.time()
    n = 0
    sig_time = 0.0
    while time.time() - t0 < secs:
        sess.run([out_name], {in_name: x})
        n += 1
        sig_time += T / sr
    wall = time.time() - t0
    rtf = wall / max(sig_time, 1e-9)
    print(f"[bench] iters={n}  wall={wall:.2f}s  signal={sig_time:.2f}s  RTF={rtf:.3f}")


def _parity(onnx_path: str, cfg_path: str, T: int = 16000, sr: int = 16000):
    # Asigură importul se_cli fără PYTHONPATH:
    root = Path(__file__).resolve().parents[1]  # proiect root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import importlib, torch
    from se_cli.config import load_config

    cfg = load_config(cfg_path, {})
    modp = cfg.get("model", {}).get("module", "se_models.mamba_unet.model:build_model")
    if ":" in modp:
        pkg, fn = modp.split(":", 1)
        mod = importlib.import_module(pkg)
        builder = getattr(mod, fn)
    else:
        mod = importlib.import_module(modp)
        builder = getattr(mod, "build_model", None) or getattr(mod, "Net", None)
        if builder is None:
            raise SystemExit(f"[parity] Nu găsesc builder în {modp}")

    try:
        net = builder(cfg)
    except TypeError:
        net = builder()

    net.eval().cpu()
    x = np.random.randn(1, 1, T).astype(np.float32)
    with torch.no_grad():
        y_t = net(torch.from_numpy(x)).cpu().numpy()
    if y_t.ndim == 2:
        y_t = y_t[:, None, :]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name, out_name = _guess_in_out(sess, _load_sidecar(Path(onnx_path)))
    y_o = sess.run([out_name], {in_name: x})[0]
    if y_o.ndim == 2:
        y_o = y_o[:, None, :]

    l2 = np.linalg.norm(y_t - y_o) / (np.linalg.norm(y_t) + 1e-12)
    mae = np.mean(np.abs(y_t - y_o))
    print(f"[parity] L2-rel={l2:.3e}  MAE={mae:.3e}  (tol target: L2<=1e-2)")
    print("[parity] OK" if l2 < 1e-2 else "[parity] WARN: paritatea e slabă (opset/dynamic/padding).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="Model ONNX")
    ap.add_argument("--config", help="YAML (pentru paritate)", default=None)
    ap.add_argument("--bench", action="store_true", help="Rulează un mic benchmark CPU (RTF)")
    ap.add_argument("--T", type=int, default=16000, help="Lungime input (mostre)")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = ap.parse_args()

    p = str(Path(args.onnx).expanduser())
    print(f"[check] file: {p}  size={Path(p).stat().st_size / 1e6:.2f} MB  sha256={_sha256(Path(p))[:16]}...")
    _check(p)
    p2 = _simplify(p)
    sess = _run_once(p2, T=args.T)
    if args.config:
        _parity(p2, args.config, T=args.T, sr=args.sr)
    if args.bench:
        _bench(sess, secs=5.0, T=args.T, sr=args.sr)


if __name__ == "__main__":
    main()
