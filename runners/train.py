from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from se_cli.utils.seed import set_random_seeds

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_text(self, *a, **k): pass
        def flush(self): pass
        def close(self): pass

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_name = cfg.get("model",{}).get("name","mamba_unet")
    module = __import__(f"se_models.{model_name}.model", fromlist=["*"])
    return module.build_model(cfg)

def _dummy_loader(n=128, sr=16000, secs=1, batch=8):
    T = sr * secs
    x = torch.randn(n, 1, T)
    y = x * 0.8  # pretend clean is 0.8*x
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch, shuffle=True)

def main(cfg: Dict[str, Any]):
    set_random_seeds(1234)
    out_root = Path(cfg.get("experiment",{}).get("out_dir", "artifacts/exp"))
    run_dir = _ensure_dir(out_root / (cfg.get("experiment",{}).get("name","mamba_unet_v0")) / time.strftime("%Y%m%d-%H%M%S"))
    ckpt_dir = _ensure_dir(run_dir / "ckpt")
    logs_dir = _ensure_dir(run_dir / "logs")
    writer = SummaryWriter(str(logs_dir))

    # data
    sr = int(cfg.get("data",{}).get("sample_rate",16000))
    batch = int(cfg.get("train",{}).get("batch_size",8))
    epochs = int(cfg.get("train",{}).get("epochs",3))

    loader = _dummy_loader(n=64, sr=sr, secs=1, batch=batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train",{}).get("lr",1e-3)))
    loss_fn = nn.L1Loss()

    train_log = {"events": []}

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0; num = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            if pred.dim()==2: pred = pred.unsqueeze(1)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            total += float(loss.item()); num += 1
        avg = total/max(1,num)
        writer.add_scalar("train/loss", avg, ep)
        train_log["events"].append({"epoch": ep, "loss": avg})

    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(model.state_dict(), ckpt_path)
    writer.add_text("checkpoint/path", str(ckpt_path), epochs)
    writer.flush(); writer.close()

    (ckpt_dir / "DONE").write_text(time.asctime(), encoding="utf-8")
    with open(run_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"[train] saved: {ckpt_path} | runs: {run_dir}")