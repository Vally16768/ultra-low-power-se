#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_name = cfg.get("model",{}).get("name","mamba_unet")
    module = __import__(f"se_models.{model_name}.model", fromlist=["*"])
    return module.build_model(cfg)


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

    # I/O names & dynamic axes from config (will be reconciled with actual signature)
    default_name = cfg.get('experiment', {}).get('name', 'model')
    path = Path(onnx_cfg.get("path", f"artifacts/export/{default_name}.onnx"))
    path.parent.mkdir(parents=True, exist_ok=True)
    opset = int(opset_override if opset_override is not None else onnx_cfg.get("opset", 18))

    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=[in_name], output_names=[out_name],
        dynamic_axes={in_name: {2: "T"}, out_name: {1: "T"}},  # output is (B,T)
        opset_version=opset, do_constant_folding=True
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
    ap.add_argument("--opset", type=int, default=None, help="override opset_version pentru export (ex: 17)")
    ap.add_argument("--skip-parity", action="store_true", help="nu rula testul de paritate")
    ap.add_argument("--frames", type=int, default=1000, help="numărul de cadre pt. paritate")
    ap.add_argument("--tol", type=float, default=1e-5, help="toleranța L2")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    onnx_path = export_onnx(cfg, opset_override=args.opset)

    if not args.skip_parity:
        parity_test(cfg, onnx_path, n_frames=args.frames, tol=args.tol)


if __name__ == "__main__":
    main()
