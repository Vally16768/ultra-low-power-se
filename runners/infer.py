#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any
import soundfile as sf
import torch

from se_core.modeling import build_model_from_cfg

@torch.no_grad()
def enhance_file(model, wav_path: str, out_path: str, sr: int):
    x, rsr = sf.read(wav_path, dtype="float32")
    if rsr != sr: raise RuntimeError(f"SR mismatch: {rsr} vs {sr}")
    if x.ndim==2: x=x.mean(axis=1)
    xt = torch.from_numpy(x).float().to(next(model.parameters()).device).unsqueeze(0).unsqueeze(1)
    y = model(xt).squeeze().cpu().numpy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr)

def main(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg).to(device)
    ckpt = cfg["inference"].get("checkpoint") or str(Path(cfg["train"]["outdir"])/"checkpoints"/"best.ckpt")
    sd = torch.load(ckpt, map_location="cpu")["model"]; model.load_state_dict(sd, strict=True); model.eval()

    sr = cfg["data"]["sample_rate"]
    in_wav = cfg["inference"].get("in_wav")
    if in_wav:
        out_wav = cfg["inference"].get("out_wav", "enhanced.wav")
        enhance_file(model, in_wav, out_wav, sr); print(f"[enhance] wrote {out_wav}"); return

    import csv
    pairs_csv = cfg["data"]["test_csv"]
    outdir = Path(cfg["inference"].get("outdir", "artifacts/eval/max/enhanced")); outdir.mkdir(parents=True, exist_ok=True)
    with open(pairs_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noisy = row["noisy"]; fname = Path(noisy).stem + "_enh.wav"
            enhance_file(model, noisy, str(outdir/fname), sr)
    print(f"[enhance] wrote enhanced wavs to {outdir}")
