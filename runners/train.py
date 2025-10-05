#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from se_core.modeling import build_model_from_cfg
from se_core.datasets import Pairset, collate
from se_core.losses import MultiResSTFTLoss, sisdr_loss, dc_offset_loss

def main(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(cfg["train"]["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(outdir/"tb"))

    sr = cfg["data"]["sample_rate"]; seg = cfg["data"].get("segment_seconds", 2.0)
    tr_ds = Pairset(cfg["data"]["train_csv"], sr, seg)
    va_ds = Pairset(cfg["data"]["val_csv"],   sr, seg)
    tr_dl = DataLoader(tr_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 0), collate_fn=collate, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers",0), collate_fn=collate, drop_last=False)

    model = build_model_from_cfg(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], betas=(0.9,0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=True)

    stft_loss = MultiResSTFTLoss(
        fft_sizes=(256,512,1024),
        hop_sizes=(64,128,256),
        win_lengths=(256,512,1024),
        mag_weight=cfg["loss"].get("mrstft_mag_w",0.5),
        sc_weight=cfg["loss"].get("mrstft_sc_w",0.5),
    )

    best = 1e9; gstep=0; epochs=cfg["train"]["epochs"]; log_every=cfg["train"].get("log_every", 50)

    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for noisy, clean in tqdm(tr_dl, ncols=100, desc=f"[train] {ep}/{epochs}"):
            noisy=noisy.to(device); clean=clean.to(device)
            opt.zero_grad()
            est = model(noisy.unsqueeze(1)).squeeze(1)
            l1 = F.l1_loss(est, clean)
            sc, mag = stft_loss(est, clean)
            sdr = sisdr_loss(est, clean)
            dc = dc_offset_loss(est)
            loss = cfg["loss"].get("l1_w",0.15)*l1 + cfg["loss"].get("mrstft_w",0.6)*(sc+mag) + cfg["loss"].get("sisdr_w",0.22)*sdr + cfg["loss"].get("dc_w",0.03)*dc
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tr_loss += loss.item()
            if gstep % log_every == 0:
                tb.add_scalar("train/total", loss.item(), gstep)
                tb.add_scalar("train/l1", l1.item(), gstep)
                tb.add_scalar("train/sisdr", sdr.item(), gstep)
                tb.add_scalar("train/mrstft", (sc+mag).item(), gstep)
                tb.add_scalar("train/dc", dc.item(), gstep)
            gstep += 1
        tr_loss /= len(tr_dl)

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for noisy, clean in va_dl:
                noisy=noisy.to(device); clean=clean.to(device)
                est = model(noisy.unsqueeze(1)).squeeze(1)
                l1 = F.l1_loss(est, clean)
                sc, mag = stft_loss(est, clean)
                sdr = sisdr_loss(est, clean)
                dc = dc_offset_loss(est)
                va_loss += (cfg["loss"].get("l1_w",0.15)*l1 + cfg["loss"].get("mrstft_w",0.6)*(sc+mag) + cfg["loss"].get("sisdr_w",0.22)*sdr + cfg["loss"].get("dc_w",0.03)*dc).item()
        va_loss /= len(va_dl); sch.step(va_loss)
        tb.add_scalar("val/total", va_loss, ep)

        ckpt_dir = outdir/"checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir/"last.ckpt")
        if va_loss < best:
            best = va_loss; torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir/"best.ckpt")
        print(f"[epoch {ep}] train={tr_loss:.4f} val={va_loss:.4f}")

    tb.close()
