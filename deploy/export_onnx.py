#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import yaml


# -------------------------
# Helpers: model loading
# -------------------------
def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Preferă se_models.<name>.build_model(cfg); dacă nu există, folosește models.factory.
    """
    name = cfg.get("model", {}).get("name", "mamba_unet")
    try:
        mod = importlib.import_module(f"se_models.{name}")
        return mod.build_model(cfg)
    except Exception:
        from models.factory import build_model_from_cfg as _factory
        return _factory(cfg)


def _load_ckpt_state_dict(ckpt_path: str, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(blob, dict) and all(isinstance(v, torch.Tensor) for v in blob.values()):
        return blob  # state_dict
    # Fallbacks (older formats)
    sd = None
    if isinstance(blob, dict):
        sd = blob.get("state_dict") or blob.get("model")
    return sd


def _infer_state_dim(model: torch.nn.Module, frame: int, device: torch.device) -> Tuple[bool, int]:
    """
    Rulează un forward scurt și încearcă să detecteze dacă modelul e stateful și care e dimensiunea stării.
    Returnează (is_stateful, state_dim).
    """
    x = torch.randn(1, 1, frame, device=device)  # [B,1,T]
    model.eval()
    with torch.no_grad():
        try:
            y = model(x)
            if isinstance(y, (tuple, list)) and len(y) == 2:
                y0, h = y
                h = h if isinstance(h, torch.Tensor) else torch.as_tensor(h)
                return True, int(h.shape[-1])
            else:
                return False, 0
        except TypeError:
            # poate cere state la input
            try:
                for hdim in (32, 64, 128, 256, 512):
                    h_in = torch.zeros(1, hdim, device=device)
                    y = model(x, h_in)
                    if isinstance(y, (tuple, list)) and len(y) == 2:
                        return True, hdim
                    else:
                        return True, hdim
            except Exception:
                pass
    return False, 0


# -------------------------
# Wrapper: force [B,1,T] output
# -------------------------
class KeepChannelWrapper(torch.nn.Module):
    """
    Înfășoară modelul astfel încât ieșirea principală să fie întotdeauna [B,1,T].
    Dacă modelul returnează [B,T], se adaugă canalul cu unsqueeze(1).
    Dacă modelul e stateful și întoarce (y, h_out), ajustează doar y.
    """
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, *state):
        out = self.net(x, *state) if state else self.net(x)
        if isinstance(out, torch.Tensor):
            y = out
            if y.dim() == 2:      # [B,T] -> [B,1,T]
                y = y.unsqueeze(1)
            return y
        elif isinstance(out, (tuple, list)) and len(out) >= 1:
            y = out[0]
            rest = list(out[1:])
            if isinstance(y, torch.Tensor) and y.dim() == 2:
                y = y.unsqueeze(1)
            return (y, *rest)
        return out


# -------------------------
# Export logic
# -------------------------
def export_onnx(cfg: Dict[str, Any]) -> Path:
    onnx_cfg = cfg.get("export", {}).get("onnx", {})
    sr = int(cfg.get("data", {}).get("sample_rate", 16000))
    frame_ms = int(cfg.get("inference", {}).get("frame_ms", 20))
    T = max(sr * frame_ms // 1000, 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = build_model_from_cfg(cfg).to(device)

    # Load weights (pure state_dict recommended)
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint", {}).get("dir", "artifacts/checkpoints/exp"))
    ckpt_path = str(ckpt_dir / "model.ckpt")
    state_dict = _load_ckpt_state_dict(ckpt_path, device)
    if state_dict:
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[export] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    base_model.eval()

    # Detect statefulness on the base model
    is_stateful, hdim = _infer_state_dim(base_model, T, device)
    print(f"[export] stateful={is_stateful}  hdim={hdim}")

    # Wrap model to ensure [B,1,T] output for 'enhanced'
    model = KeepChannelWrapper(base_model).to(device).eval()

    # I/O names & dynamic axes from config (will be reconciled with actual signature)
    path = Path(onnx_cfg.get("path", f"artifacts/export/{cfg.get('experiment',{}).get('name','model')}.onnx"))
    path.parent.mkdir(parents=True, exist_ok=True)
    opset = int(onnx_cfg.get("opset", 18))

    # Desired (from config), but may contain h_in/h_out
    inputs_cfg: List[Dict[str, str]]  = onnx_cfg.get("inputs",  [{"name": "noisy"}] + ([{"name": "h_in"}]  if is_stateful else []))
    outputs_cfg: List[Dict[str, str]] = onnx_cfg.get("outputs", [{"name": "enhanced"}] + ([{"name": "h_out"}] if is_stateful else []))

    # Example inputs
    noisy = torch.randn(1, 1, T, device=device)
    args = [noisy]
    if is_stateful:
        h_in = torch.zeros(1, hdim, device=device)
        args.append(h_in)

    # Dry run — get actual number of outputs on the WRAPPED model
    with torch.no_grad():
        out = model(*args)

    # Normalize to (tuple of tensors)
    if isinstance(out, torch.Tensor):
        out_tuple = (out,)
        actual_stateful = False
    elif isinstance(out, (tuple, list)):
        out_tuple = tuple(out)
        actual_stateful = (len(out_tuple) == 2)
    else:
        raise RuntimeError("Model forward returned unsupported type")

    # Reconcile inputs/outputs with actual signature
    in_names  = [i["name"] for i in inputs_cfg]
    out_names = [o["name"] for o in outputs_cfg]

    if not actual_stateful:
        # keep only the noisy input
        args = [noisy]
        in_names = ["noisy"]
        # keep only the enhanced output
        out_names = ["enhanced"]
    else:
        # model returns two outputs; ensure we have exactly 2 names
        if "enhanced" not in out_names:
            out_names = ["enhanced"] + [n for n in out_names if n != "enhanced"]
        if len(out_names) < 2:
            out_names = ["enhanced", "h_out"]
        out_names = out_names[:2]
        # ensure h_in present in inputs
        if "h_in" not in in_names:
            in_names = ["noisy", "h_in"]

    # Dynamic axes — rebuild to match FINAL names and expected ranks
    dyn_axes = onnx_cfg.get("dynamic_axes", {})
    dyn_axes = {k: v for k, v in dyn_axes.items() if k in set(in_names) | set(out_names)}

    # Input is [B,1,T]
    dyn_axes.setdefault("noisy", {0: "B", 2: "T"})

    # Output 'enhanced' is enforced to [B,1,T] by wrapper
    dyn_axes["enhanced"] = {0: "B", 2: "T"}

    if actual_stateful:
        dyn_axes.setdefault("h_in",  {0: "B"})
        dyn_axes.setdefault("h_out", {0: "B"})

    # Export
    torch.onnx.export(
        model,
        tuple(args),
        f=str(path),
        input_names=in_names,
        output_names=out_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dyn_axes,
    )
    print(f"[export] ONNX saved to: {path}")

    # Save small sidecar with info (aligned with dynamic_axes used)
    sidecar = {
        "stateful": bool(actual_stateful),
        "state_dim": int(hdim if actual_stateful else 0),
        "sr": sr,
        "frame_ms": frame_ms,
        "inputs": in_names,
        "outputs": out_names,
        "dynamic_axes": dyn_axes,
        "opset": opset,
    }
    Path(str(path) + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


# -------------------------
# Parity test (streaming)
# -------------------------
def _to_np(y: torch.Tensor) -> np.ndarray:
    """
    Acceptă [B,1,T] sau [B,T] și întoarce [T].
    """
    if y.dim() == 3:  # [B,1,T]
        y = y[:, 0, :]
    return y.squeeze(0).detach().cpu().numpy()


def parity_test(cfg: Dict[str, Any], onnx_path: Path, n_frames: int = 1000, tol: float = 1e-5):
    import onnxruntime as ort

    info = json.loads(Path(str(onnx_path) + ".json").read_text(encoding="utf-8"))
    is_stateful = bool(info["stateful"])
    sr = int(info["sr"])
    frame_ms = int(info["frame_ms"])
    T = max(sr * frame_ms // 1000, 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = build_model_from_cfg(cfg).to(device)
    # load weights for base_model
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint", {}).get("dir", "artifacts/checkpoints/exp"))
    ckpt_path = str(ckpt_dir / "model.ckpt")
    sd = _load_ckpt_state_dict(ckpt_path, device)
    if sd:
        base_model.load_state_dict(sd, strict=False)
    base_model.eval()

    # Wrap for [B,1,T] parity identical to export
    model = KeepChannelWrapper(base_model).to(device).eval()

    # ORT session
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    noisy_name = "noisy" if "noisy" in in_names else in_names[0]
    enhanced_name = "enhanced" if "enhanced" in out_names else out_names[0]
    h_in_name = "h_in" if "h_in" in in_names else (in_names[1] if is_stateful and len(in_names) > 1 else None)
    h_out_name = "h_out" if "h_out" in out_names else (out_names[1] if is_stateful and len(out_names) > 1 else None)

    max_l2 = 0.0
    state_pt = None
    state_onnx = None
    rng = np.random.default_rng(1337)

    with torch.no_grad():
        for _ in range(n_frames):
            x = rng.standard_normal(T).astype(np.float32)
            xb = torch.from_numpy(x).to(device).view(1, 1, -1)  # [B,1,T]

            # PyTorch fwd on wrapped model
            try:
                if is_stateful:
                    if state_pt is None and h_in_name:
                        state_pt = torch.zeros(1, int(info["state_dim"]), device=device)
                    y_pt, state_pt = model(xb, state_pt)
                else:
                    y_pt = model(xb)
            except TypeError:
                if state_pt is None and is_stateful:
                    state_pt = torch.zeros(1, int(info["state_dim"]), device=device)
                y_pt, state_pt = model(xb, state_pt)

            y_pt = _to_np(y_pt)  # [T]

            # ONNX fwd
            feed = {noisy_name: xb.cpu().numpy()}
            if is_stateful and h_in_name:
                if state_onnx is None:
                    state_onnx = np.zeros((1, int(info["state_dim"])), dtype=np.float32)
                feed[h_in_name] = state_onnx
            out = sess.run(None, feed)
            if is_stateful and h_out_name:
                y_ox = out[out_names.index(enhanced_name)]
                state_onnx = out[out_names.index(h_out_name)]
            else:
                y_ox = out[out_names.index(enhanced_name)]
            # y_ox expected [B,1,T] → [T]
            y_ox = np.asarray(y_ox).squeeze()
            if y_ox.ndim == 2:  # [1,T]
                y_ox = y_ox[0]
            elif y_ox.ndim == 3:  # [1,1,T]
                y_ox = y_ox[0, 0]

            l2 = float(np.linalg.norm(y_pt - y_ox, ord=2))
            if l2 > max_l2:
                max_l2 = l2
            if l2 > tol:
                raise AssertionError(f"[parity] L2 error {l2:.3e} exceeded tol={tol:.1e}")

    print(f"[parity] OK. max L2 over {n_frames} frames: {max_l2:.3e} (tol={tol:.1e})")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/<exp>.yaml")
    ap.add_argument("--skip-parity", action="store_true", help="nu rula testul de paritate")
    ap.add_argument("--frames", type=int, default=1000, help="numărul de cadre pt. paritate")
    ap.add_argument("--tol", type=float, default=1e-5, help="toleranța L2")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    onnx_path = export_onnx(cfg)

    if not args.skip_parity:
        parity_test(cfg, onnx_path, n_frames=args.frames, tol=args.tol)


if __name__ == "__main__":
    main()
