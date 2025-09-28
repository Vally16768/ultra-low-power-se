# deploy/prune.py
import argparse, os, sys
from pathlib import Path

def seed_all(seed=404):
    import torch, random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_factory(factory_arg=None):
    from importlib import import_module
    candidates = []
    if factory_arg:
        if ":" not in factory_arg:
            raise SystemExit("Format factory invalid. Folosește module:callable (ex: se_models.mamba_unet.model:build_model)")
        candidates.append(tuple(factory_arg.split(":",1)))
    candidates += [
        ("se_models.mamba_unet.model","build_model"),
        ("se_models.mamba_unet.model","make_model"),
        ("se_models.mamba_unet.model","create_model"),
        ("se_models.mamba_unet.model","Model"),
        ("se_models.mamba_unet.model","MambaUNet"),
    ]
    for mod, name in candidates:
        try:
            m = import_module(mod); fn = getattr(m, name)
            return fn, f"{mod}:{name}"
        except Exception:
            continue
    raise ImportError("Nu am găsit un factory pentru model. Specifică-l cu --factory module:callable")

def construct_model(factory_fn, config_path):
    import yaml, inspect
    cfg = None
    if config_path:
        cfg = yaml.safe_load(Path(config_path).read_text())
    if inspect.isclass(factory_fn):
        if cfg is not None and hasattr(factory_fn, "from_config"):
            return factory_fn.from_config(cfg)
        if cfg is not None:
            try: return factory_fn(**cfg)
            except Exception: pass
        return factory_fn()
    sig = inspect.signature(factory_fn)
    try:
        if cfg is not None and len(sig.parameters)==1:
            return factory_fn(cfg)
        if cfg is not None:
            return factory_fn(**cfg)
        return factory_fn()
    except Exception:
        return factory_fn()

def iter_conv_modules(model, scope):
    import torch.nn as nn
    for n,m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if not scope or any(s in n for s in scope):
                yield n,m

def l2_structured_prune(model, pct=0.3, scope=("encoder","decoder")):
    from torch.nn.utils import prune
    count = 0
    for n,m in iter_conv_modules(model, scope):
        prune.ln_structured(m, name="weight", amount=pct, n=2, dim=0)
        prune.remove(m, "weight")
        count += 1
    return count

def finetune_short(model, epochs=2, device=None):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy = torch.randn(8,1,32000, device=device)
    for _ in range(epochs):
        opt.zero_grad(); y = model(dummy); loss = (y**2).mean(); loss.backward(); opt.step()
    model.eval().to("cpu")
    return model

def export_to_onnx(model, out):
    import torch
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1,1,32000)
    torch.onnx.export(model, dummy, out, opset_version=18,
                      input_names=["noisy"], output_names=["enhanced"],
                      dynamic_axes={"noisy":{0:"B",2:"T"}, "enhanced":{0:"B",2:"T"}})

def safe_load_ckpt(path):
    import torch
    try:
        # PyTorch ≥2.4: reduce riscul pickle
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ckpt", required=True)
    ap.add_argument("--config", required=False)
    ap.add_argument("--factory", required=False)
    ap.add_argument("--pct", type=float, default=0.3)
    ap.add_argument("--scope", type=str, default="encoder,decoder")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--export-onnx", type=str)
    args = ap.parse_args()

    if not Path(args.train_ckpt).exists():
        raise SystemExit(f"[ERROR] checkpoint inexistent: {args.train_ckpt}")

    seed_all()
    factory_fn, factory_name = resolve_factory(args.factory)
    model = construct_model(factory_fn, args.config)

    sd = safe_load_ckpt(args.train_ckpt)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)

    scope = tuple([s.strip() for s in args.scope.split(",") if s.strip()])
    nmods = l2_structured_prune(model, pct=args.pct, scope=scope)
    print(f"[prune] factory={factory_name}  modules_pruned={nmods}  amount={args.pct*100:.0f}%  scope={scope}")

    model = finetune_short(model, epochs=args.epochs)

    Path(os.path.dirname(args.out_ckpt) or ".").mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"model": model.state_dict()}, args.out_ckpt)
    print(f"[save] {args.out_ckpt}")

    if args.export_onnx:
        export_to_onnx(model, args.export_onnx)
        print(f"[export] {args.export_onnx}")

if __name__ == "__main__":
    main()
