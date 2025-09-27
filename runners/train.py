from __future__ import annotations
import os, json, time, importlib
from pathlib import Path
from typing import Dict, Any
import torch
from se_cli.utils.seed import set_random_seeds

# sus, în locul importului direct
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_text(self, *a, **k): pass
        def flush(self): pass
        def close(self): pass


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _build_model(cfg: Dict[str, Any]):
    name = cfg.get("model", {}).get("name", "mamba_unet")
    mod = importlib.import_module(f"se_models.{name}")
    model = mod.build_model(cfg)
    return model

def _save_model_card(out_dir: Path, cfg: Dict[str, Any], params_count: int):
    card = out_dir / "card.md"
    lines = [
        f"# Model Card — {cfg.get('experiment',{}).get('name','exp')}",
        "",
        "## Summary",
        "- Task: Speech Enhancement (low-power, streaming)",
        f"- Params: **{params_count:,}**",
        f"- Causal: **{cfg.get('model',{}).get('causal', True)}**, Lookahead: **{cfg.get('model',{}).get('lookahead_ms', 0)} ms**",
        "",
        "## Data",
        f"- Train manifest: `{cfg.get('data',{}).get('train_manifest','')}`",
        f"- Dev manifest: `{cfg.get('data',{}).get('dev_manifest','')}`",
        "",
        "## Training",
        f"- Optimizer: `{cfg.get('train',{}).get('optimizer','adamw')}`, LR: `{cfg.get('train',{}).get('lr',1e-3)}`",
        f"- Epochs: `{cfg.get('train',{}).get('epochs',1)}`",
        "",
        "## Export",
        f"- ONNX opset: `{cfg.get('export',{}).get('onnx',{}).get('opset','')}`",
        "",
        "## Notes",
        cfg.get('experiment',{}).get('notes',''),
        ""
    ]
    card.write_text("\n".join(lines), encoding="utf-8")

def main(cfg: Dict[str, Any]):
    exp = cfg.get("experiment", {}).get("name", "exp")
    seed = int(cfg.get("experiment", {}).get("seed", cfg.get("seed", 1337)))
    set_random_seeds(seed)

    # output dirs
    ckpt_dir = _ensure_dir(cfg.get("train", {}).get("checkpoint", {}).get("dir", f"artifacts/checkpoints/{exp}"))
    run_dir  = _ensure_dir(f"artifacts/runs/{exp}")
    log_json = ckpt_dir / "train_log.json"

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _save_model_card(ckpt_dir, cfg, n_params)

    # optimizer/scheduler (schelet)
    opt_name = cfg.get("train", {}).get("optimizer", "adamw").lower()
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if opt_name=="adamw" else torch.optim.Adam(model.parameters(), lr=lr)

    # dummy loop (dacă nu ai încă dataloader-ul): efectuează un forward pe zgomot aleator
    writer = SummaryWriter(str(run_dir))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    writer.add_text("model/summary", f"params: {n_params:,}", 0)

    train_log = {"start_time": time.asctime(), "epochs": epochs, "events": []}

    for ep in range(1, epochs+1):
        model.train()
        # batch dummy: 1s @16k
        sr = int(cfg.get("data",{}).get("sample_rate", 16000))
        x = torch.randn(4, sr, device=device)
        y = model(x)  # stub forward

        loss = (y - x).abs().mean()  # L1 dummy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("train",{}).get("grad_clip", 5.0)))
        optimizer.step()

        writer.add_scalar("train/loss", float(loss.item()), ep)
        train_log["events"].append({"epoch": ep, "loss": float(loss.item())})

    # save checkpoint (weights + config slim)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(model.state_dict(), ckpt_path)
    writer.add_text("checkpoint/path", str(ckpt_path), epochs)
    writer.flush(); writer.close()

    (ckpt_dir / "DONE").write_text(time.asctime(), encoding="utf-8")
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"[train] saved: {ckpt_path} | log: {log_json} | runs: {run_dir}")
