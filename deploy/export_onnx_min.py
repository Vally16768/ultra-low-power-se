#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, importlib, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

def load_symbol(path: str):
    if ":" not in path:
        raise ValueError("Folosește module:attr (ex: se_models.mamba_unet.model:build_model)")
    mod, attr = path.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, attr)

def load_cfg(yaml_path: str | None) -> Dict[str, Any]:
    if not yaml_path:
        return {}
    try:
        import yaml
    except Exception:
        print("[warn] PyYAML nu e instalat; ignor --config", file=sys.stderr)
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_model(symbol, cfg: Dict[str, Any]) -> torch.nn.Module:
    if callable(symbol):
        try:
            return symbol(cfg)
        except Exception:
            try:
                return symbol()
            except Exception as e:
                raise RuntimeError(f"Nu pot construi modelul: {e}") from e
    if hasattr(symbol, "build_model"):
        return symbol.build_model(cfg)
    if hasattr(symbol, "Net"):
        try:
            return symbol.Net(cfg)
        except Exception:
            return symbol.Net()
    raise RuntimeError("Simbolul dat nu e apelabil și nu are build_model/Net")

def make_dummy(input_shape: str | None, sr: int, seconds: float, device: torch.device) -> torch.Tensor:
    if input_shape:
        dims = [int(x) for x in input_shape.strip().split(",")]
        return torch.randn(*dims, device=device, dtype=torch.float32)
    base = 320 if sr == 16000 else max(1, sr // 50)  # ~20 ms
    T = max(base, int(round(sr * seconds / base) * base))
    return torch.randn(1, 1, T, device=device, dtype=torch.float32)  # [B,1,T]

def infer_outputs(model: torch.nn.Module, x: torch.Tensor) -> Tuple[List[str], List[torch.Tensor]]:
    model.eval()
    with torch.no_grad():
        y = model(x)
    if isinstance(y, torch.Tensor):
        return ["enhanced"], [y]
    if isinstance(y, (tuple, list)):
        names = [f"output{i}" for i in range(len(y))]
        ts: List[torch.Tensor] = []
        for t in y:
            if not isinstance(t, torch.Tensor):
                raise RuntimeError("Output non-Tensor într-un tuple/list.")
            ts.append(t)
        return names, ts
    raise RuntimeError("Modelul a returnat tip ne-suportat (aștept Tensor sau list/tuple de Tensor).")

def export_classic(model, x, out_path: Path, opset: int, in_name: str, out_names: List[str], dynamic: bool) -> None:
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {in_name: {-1: "T"}}
        for n in out_names:
            dynamic_axes[n] = {-1: "T"}
    torch.onnx.export(
        model, x, str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        input_names=[in_name],
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False,
    )

def try_dynamo(model, x, out_path: Path, dynamic: bool) -> bool:
    try:
        from torch.onnx import dynamo_export
    except Exception:
        return False
    try:
        ep = dynamo_export(model.eval(), x, dynamic_shapes=bool(dynamic))
        ep.save(str(out_path), external_data=False)
        print(f"[ok] dynamo_export → {out_path}")
        return True
    except Exception as e:
        print(f"[info] dynamo_export a eșuat: {e!r}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Export ONNX (automat, simplu).")
    ap.add_argument("--model", required=True, help="module:attr (ex: se_models.mamba_unet.model:build_model)")
    ap.add_argument("--config", default=None, help="YAML opțional (trece ca dict la builder)")
    ap.add_argument("--checkpoint", default=None, help="ckpt .pt/.pth opțional (CPU map)")
    ap.add_argument("--out", default="artifacts/export/model.onnx", help="cale ONNX rezultat")
    ap.add_argument("--opset", type=int, default=17, help="opset pt. export clasic (default 17)")
    ap.add_argument("--use-dynamo", type=int, default=0, help="1 = încearcă întâi dynamo_export (opset 18)")
    ap.add_argument("--dynamic", type=int, default=0, help="1 = dimensiune temporală dinamică pe ultima axă")
    ap.add_argument("--input-shape", default=None, help="ex: 1,1,16000; altfel se folosește sr/seconds")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--input-name", default="noisy")  # aliniat cu verify
    ap.add_argument("--keep-ch", type=int, default=1, help="1 = forțează ieșire [B,1,T]")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = torch.device("cpu")

    cfg = load_cfg(args.config)
    sym = load_symbol(args.model)
    model = build_model(sym, cfg).to(device).eval()

    # wrapper opțional pentru [B,1,T]
    if args.keep_ch:
        class KeepCh(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                y = self.net(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if isinstance(y, torch.Tensor) and y.dim() == 2:
                    y = y.unsqueeze(1)  # [B,T] -> [B,1,T]
                return y
        model = KeepCh(model).eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)

    x = make_dummy(args.input_shape, args.sr, args.seconds, device)
    out_names, _ = infer_outputs(model, x)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_dynamo:
        ok = try_dynamo(model, x, out_path, dynamic=bool(args.dynamic))
        if not ok:
            print("[info] continui cu exporterul clasic…")
            try:
                export_classic(model, x, out_path, args.opset, args.input_name, out_names, dynamic=bool(args.dynamic))
            except Exception as e_dyn:
                print(f"[info] classic dinamic a eșuat ({e_dyn!r}); încerc STATIC…")
                export_classic(model, x, out_path, args.opset, args.input_name, out_names, dynamic=False)
    else:
        try:
            export_classic(model, x, out_path, args.opset, args.input_name, out_names, dynamic=bool(args.dynamic))
        except Exception as e:
            if args.dynamic:
                print(f"[info] classic dinamic a eșuat ({e!r}); încerc STATIC…")
                export_classic(model, x, out_path, args.opset, args.input_name, out_names, dynamic=False)
            else:
                raise

    # validare minimală ONNX
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        print(f"[ok] onnx.checker: valid → {out_path}")
    except Exception as e:
        print(f"[warn] onnx.checker a raportat o problemă: {e!r}")

    side = {
        "engine": "dynamo_export" if args.use_dynamo else "classic",
        "dynamic": bool(args.dynamic),
        "opset": 18 if args.use_dynamo else args.opset,
        "input_name": args.input_name,
        "output_names": out_names,
        "example_input_shape": list(x.shape),
        "sr": int(cfg.get("data", {}).get("sample_rate", 16000)),
    }
    Path(str(out_path) + ".json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(f"[done] saved: {out_path} (+ sidecar .json)")

if __name__ == "__main__":
    main()
