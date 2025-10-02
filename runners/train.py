# runners/train.py
#!/usr/bin/env python3
"""
Trainer pentru Ultra-Low-Power SE.
- Primește 'cfg' din se_cli.cli (nu folosește argparse).
- Citește manifest-urile (CSV cu coloane: noisy, clean).
- Pierderi: L1 (time) + MR-STFT (mag L1 + spectral convergence) + SI-SDR (maximizat).
- Adaptive LR: ReduceLROnPlateau (config din YAML).
- Early Stopping pe val_loss (config din YAML).
- Checkpoint: last.ckpt, best.ckpt; TensorBoard în <outdir>/tb.
- Import model: model.module (YAML) / MODEL_MODULE (env) -> model.py (root) -> se_models.mamba_unet.model -> U-Net 1D fallback.
"""

import os
import csv
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ------------------ Utilitare ------------------
def _set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _load_wav_mono16k(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(-1)
    if sr != target_sr:
        import resampy  # lazy import
        x = resampy.resample(x, sr, target_sr)
    return x


def _si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Returnează SI-SDR pe batch: pred/target [B,T] -> [B]."""
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    s_target = (torch.sum(pred * target, dim=-1, keepdim=True)
                / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target
    e_noise = pred - s_target
    num = torch.sum(s_target ** 2, dim=-1)
    den = torch.sum(e_noise ** 2, dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """x: [B,T] -> |STFT|: [B, F, T'] """
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True)
    return torch.abs(X)


def _mrstft_loss(pred: torch.Tensor, target: torch.Tensor,
                 cfg: Tuple[Tuple[int, int, int], ...] = ((256, 64, 256),
                                                         (512, 128, 512),
                                                         (1024, 256, 1024))) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-Resolution STFT loss:
      - mag_l1: L1 pe magnitudini
      - sc: spectral convergence (||Δ||_F / ||target||_F), mediat pe batch și rezoluții
    """
    mag_l1 = 0.0
    sc = 0.0
    for (nfft, hop, win) in cfg:
        Mp = _stft_mag(pred, nfft, hop, win)   # [B,F,T]
        Mt = _stft_mag(target, nfft, hop, win) # [B,F,T]

        mag_l1 += (Mp - Mt).abs().mean()

        num = torch.linalg.norm(Mp - Mt, ord='fro', dim=(-2, -1))  # [B]
        den = torch.linalg.norm(Mt,      ord='fro', dim=(-2, -1))  # [B]
        sc  += (num / (den + 1e-8)).mean()

    mag_l1 /= len(cfg)
    sc     /= len(cfg)
    return mag_l1, sc


# ------------------ Dataset ------------------
class _PairsDataset(Dataset):
    """
    Manifest CSV cu coloane: noisy, clean (căi absolute/relative).
    Opțional: segment_len (în sample) pentru crop/pad.
    """
    def __init__(self, manifest_csv: str | Path, segment_len: int = 32000,
                 sr: int = 16000, random_crop: bool = True):
        self.items: list[tuple[str, str]] = []
        self.sr = sr
        self.seg = segment_len
        self.random_crop = random_crop

        with open(manifest_csv, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                noisy = (row.get("noisy") or "").strip()
                clean = (row.get("clean") or "").strip()
                if noisy and clean and Path(noisy).exists() and Path(clean).exists():
                    self.items.append((noisy, clean))

        if not self.items:
            raise RuntimeError(f"Empty/invalid manifest: {manifest_csv}")

        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get(self, p: str) -> np.ndarray:
        if p in self._cache:
            return self._cache[p]
        x = _load_wav_mono16k(p, self.sr)
        self._cache[p] = x
        if len(self._cache) > 256:  # bound cache
            self._cache.pop(next(iter(self._cache)))
        return x

    def __getitem__(self, idx: int):
        noisy_p, clean_p = self.items[idx]
        n = self._get(noisy_p)
        c = self._get(clean_p)

        L = min(len(n), len(c))
        n = n[:L]
        c = c[:L]

        if self.seg is not None:
            if L >= self.seg:
                st = np.random.randint(0, L - self.seg + 1) if self.random_crop else (L - self.seg) // 2
                n = n[st:st + self.seg]
                c = c[st:st + self.seg]
            else:
                pad = self.seg - L
                n = np.pad(n, (0, pad))
                c = np.pad(c, (0, pad))

        # ușor gain-jitter
        g = np.random.uniform(0.9, 1.1)
        n = (n * g).astype(np.float32)
        c = c.astype(np.float32)

        return torch.from_numpy(n).unsqueeze(0), torch.from_numpy(c).unsqueeze(0)  # [1,T], [1,T]


def _collate_pad(batch):
    Ns, Cs = zip(*batch)
    T = max(t.shape[-1] for t in Ns)
    Np, Cp, lengths = [], [], []
    for n, c in zip(Ns, Cs):
        L = n.shape[-1]
        if L < T:
            pad = T - L
            n = F.pad(n, (0, pad))
            c = F.pad(c, (0, pad))
        Np.append(n)
        Cp.append(c)
        lengths.append(L)
    return torch.stack(Np, 0), torch.stack(Cp, 0), torch.tensor(lengths, dtype=torch.long)


# ------------------ Model loading ------------------
def _get_model_from_repo(cfg: Dict[str, Any]) -> nn.Module:
    """
    Ordine import:
      0) cfg['model']['module'] (dacă e setat)
      1) MODEL_MODULE din env (e.g., se_models.mamba_unet.model)
      2) model.py din root repo (build_model(cfg) sau TinyMambaUNetStub)
      3) se_models.mamba_unet.model (dacă există)
      4) fallback SmallUNet1D
    """
    mod_name = (cfg.get("model", {}) or {}).get("module", "") \
               or os.environ.get("MODEL_MODULE", "").strip()
    user_model = None

    if mod_name:
        try:
            import importlib
            user_model = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[warn] model.module='{mod_name}' import failed: {e}")

    if user_model is None:
        try:
            import model as _m  # root
            user_model = _m
        except Exception as e:
            print(f"[warn] cannot import model.py at repo root: {e}")

    if user_model is None:
        try:
            from se_models.mamba_unet import model as _m
            user_model = _m
        except Exception as e:
            print(f"[warn] cannot import se_models.mamba_unet.model: {e} — using fallback SmallUNet1D")

    if user_model is not None:
        if hasattr(user_model, "build_model"):
            return user_model.build_model(cfg)
        if hasattr(user_model, "TinyMambaUNetStub"):
            return user_model.TinyMambaUNetStub()

    # Fallback U-Net 1D mic
    class _ConvBlock(nn.Module):
        def __init__(self, ch_in, ch_out, k=9, d=1):
            super().__init__()
            p = (k // 2) * d
            self.net = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, k, padding=p, dilation=d),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch_out, ch_out, k, padding=p, dilation=d),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):  # [B,C,T]
            return self.net(x)

    class _SmallUNet1D(nn.Module):
        def __init__(self, ch=32):
            super().__init__()
            self.enc1 = _ConvBlock(1, ch)
            self.down1 = nn.Conv1d(ch, ch * 2, 4, stride=2, padding=1)
            self.enc2 = _ConvBlock(ch * 2, ch * 2)
            self.down2 = nn.Conv1d(ch * 2, ch * 4, 4, stride=2, padding=1)
            self.enc3 = _ConvBlock(ch * 4, ch * 4, d=2)
            self.bot = _ConvBlock(ch * 4, ch * 4, d=4)
            self.up2 = nn.ConvTranspose1d(ch * 4, ch * 2, 4, stride=2, padding=1)
            self.dec2 = _ConvBlock(ch * 4, ch * 2)
            self.up1 = nn.ConvTranspose1d(ch * 2, ch, 4, stride=2, padding=1)
            self.dec1 = _ConvBlock(ch * 2, ch)
            self.out = nn.Conv1d(ch, 1, 1)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bot(e3)
            d2 = self.up2(b)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            y = self.out(d1)
            return torch.tanh(y + x)

    return _SmallUNet1D()


# ------------------ Antrenare / Validare ------------------
def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler,
               device: torch.device, train: bool, loss_cfg: Dict[str, float]) -> Tuple[float, float]:
    model.train(train)
    tot_loss = 0.0
    tot_sisdr = 0.0
    n_frames = 0

    bar = tqdm(loader, desc="train" if train else "valid", leave=False)
    for noisy, clean, lengths in bar:
        noisy = noisy.to(device)   # [B,1,T]
        clean = clean.to(device)   # [B,1,T]

        with torch.set_grad_enabled(train):
            with torch.autocast(device.type, enabled=loss_cfg["amp"]):
                pred = model(noisy).squeeze(1)   # [B,T] (model poate întoarce [B,1,T])
                clean_mono = clean.squeeze(1)    # [B,T]

                l_time = F.l1_loss(pred, clean_mono)
                l_mag, l_sc = _mrstft_loss(pred, clean_mono)
                sisdr = _si_sdr(pred, clean_mono).mean()

                # Semn minus la SI-SDR (maximizăm)
                loss = (loss_cfg["w_time"] * l_time
                        + loss_cfg["w_mag"] * l_mag
                        + loss_cfg["w_sc"] * l_sc
                        + loss_cfg["w_sisdr"] * (-sisdr / 10.0))

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip"])
                    optimizer.step()

        B = noisy.shape[0]
        tot_loss += float(loss.detach()) * B
        tot_sisdr += float(sisdr.detach()) * B
        n_frames += B

        bar.set_postfix(loss=f"{(tot_loss/max(1,n_frames)):.4f}",
                        sdr=f"{(tot_sisdr/max(1,n_frames)):.2f} dB")

    return tot_loss / max(1, n_frames), tot_sisdr / max(1, n_frames)


# ------------------ Entry Point ------------------
def main(cfg: Dict[str, Any]):
    # ---- hparams din cfg + fallback-uri ----
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    exp_cfg   = cfg.get("experiment", {}) or {}

    sr       = int(data_cfg.get("sample_rate", 16000))
    seg_sec  = float(train_cfg.get("segment_sec", 4.0))
    seg_len  = int(seg_sec * sr)
    epochs   = int(train_cfg.get("epochs", 50))
    batch    = int(train_cfg.get("batch_size", 8))
    lr       = float(train_cfg.get("lr", 2e-4))
    seed     = int(train_cfg.get("seed", 1337))

    # scheduler config (adaptive LR)
    sched_cfg = (train_cfg.get("scheduler") or {})
    sched_mode     = str(sched_cfg.get("mode", "min"))
    sched_factor   = float(sched_cfg.get("factor", 0.5))
    sched_patience = int(sched_cfg.get("patience", 5))
    sched_min_lr   = float(sched_cfg.get("min_lr", 1e-6))
    sched_cooldown = int(sched_cfg.get("cooldown", 0))
    sched_verbose  = bool(sched_cfg.get("verbose", True))

    # early stopping
    es_cfg      = (train_cfg.get("early_stopping") or {})
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta= float(es_cfg.get("min_delta", 0.0))

    out_root = Path(exp_cfg.get("out_dir", "artifacts/exp"))
    exp_name = exp_cfg.get("name", "se_experiment")
    outdir   = out_root / exp_name
    outdir.mkdir(parents=True, exist_ok=True)

    train_manifest = data_cfg.get("train_manifest", "data/prepared/train_mixes/manifests/pairs.csv")
    val_manifest   = data_cfg.get("val_manifest",   "data/prepared/dev_mixes/manifests/pairs.csv")

    # ---- seed, device, logs ----
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(str(outdir / "tb"))
    (outdir / "hparams.from_cfg.json").write_text(json.dumps(cfg, indent=2))

    # ---- model & loaders ----
    model = _get_model_from_repo(cfg).to(device)

    train_ds = _PairsDataset(train_manifest, segment_len=seg_len, sr=sr, random_crop=True)
    val_ds   = _PairsDataset(val_manifest,   segment_len=seg_len, sr=sr, random_crop=False)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=4,
                          collate_fn=_collate_pad, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4,
                          collate_fn=_collate_pad, pin_memory=True)

    # ---- opt/sched/amp ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, factor=sched_factor, patience=sched_patience,
        cooldown=sched_cooldown, min_lr=sched_min_lr, verbose=sched_verbose)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    loss_cfg: Dict[str, float] = {
        "amp": True,
        "grad_clip": 5.0,
        "w_time": 0.2,
        "w_mag": 0.6,
        "w_sc": 0.2,
        "w_sisdr": 1.0,
    }

    best_val = float("inf")
    best_path = None
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_sdr = _run_epoch(model, train_ld, optimizer, scaler, device, train=True,  loss_cfg=loss_cfg)
        va_loss, va_sdr = _run_epoch(model, val_ld,   optimizer, scaler, device, train=False, loss_cfg=loss_cfg)
        dt = time.time() - t0

        # logs
        writer.add_scalar("loss/train", tr_loss, ep)
        writer.add_scalar("loss/val",   va_loss, ep)
        writer.add_scalar("sisdr/train", tr_sdr, ep)
        writer.add_scalar("sisdr/val",   va_sdr, ep)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], ep)

        print(f"[ep {ep:03d}] "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
              f"SI-SDR tr={tr_sdr:.2f} val={va_sdr:.2f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        # adaptive LR (monitor val_loss)
        scheduler.step(va_loss)

        # checkpoint last
        last_path = outdir / "last.ckpt"
        torch.save({"model": model.state_dict(), "ep": ep, "val_loss": va_loss}, last_path)

        # checkpoint best + early stopping book-keeping
        if va_loss < (best_val - es_min_delta):
            best_val = va_loss
            best_path = outdir / "best.ckpt"
            torch.save({"model": model.state_dict(), "ep": ep, "val_loss": va_loss}, best_path)
            epochs_no_improve = 0
            print(f"  ↳ new best: {best_val:.4f} → {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  ↳ no improvement ({epochs_no_improve}/{es_patience})")

        # early stopping
        if epochs_no_improve >= es_patience:
            print(f"[early-stopping] No improvement ≥ {es_min_delta} for {es_patience} epochs. Stop.")
            break

    print(f"Done. Best val_loss={best_val:.4f} at {best_path}")
