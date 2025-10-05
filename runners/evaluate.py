#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np, soundfile as sf, torch

from se_core.modeling import build_model_from_cfg

try:
    from pesq import pesq
except Exception:
    pesq = None
try:
    from pystoi import stoi as _stoi
except Exception:
    _stoi = None

def si_sdr(ref, est, eps=1e-8):
    ref = ref - np.mean(ref); est = est - np.mean(est)
    s = np.sum(ref*est) * ref / (np.sum(ref**2) + eps)
    e = est - s
    return 10*np.log10((np.sum(s**2)+eps)/(np.sum(e**2)+eps))

def main(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg["data"]["sample_rate"]
    model = build_model_from_cfg(cfg).to(device)
    ckpt = cfg["inference"].get("checkpoint") or str(Path(cfg["train"]["outdir"])/"checkpoints"/"best.ckpt")
    sd = torch.load(ckpt, map_location="cpu")["model"]; model.load_state_dict(sd, strict=True); model.eval()

    pairs_csv = cfg["data"]["test_csv"]
    outdir = Path(cfg["eval"].get("outdir", "artifacts/eval/max/enhanced")); outdir.mkdir(parents=True, exist_ok=True)

    from runners.infer import enhance_file
    with open(pairs_csv,"r") as f: rows = list(csv.DictReader(f))
    enh_paths: List[Tuple[str,str]] = []
    for row in rows:
        noisy = row["noisy"]; clean=row["clean"]; fname = Path(noisy).stem + "_enh.wav"; out_p = str(outdir/fname)
        enhance_file(model, noisy, out_p, sr); enh_paths.append((out_p, clean))

    scores = {"PESQ": [], "STOI": [], "SI_SDR": []}
    for enh, clean in enh_paths:
        y,_ = sf.read(enh, dtype="float32"); c,_ = sf.read(clean, dtype="float32")
        L=min(len(y),len(c)); y=y[:L]; c=c[:L]
        if pesq is not None:
            try: scores["PESQ"].append(pesq(sr, c, y, "wb"))
            except Exception: pass
        if _stoi is not None:
            try: scores["STOI"].append(_stoi(c, y, sr, extended=False))
            except Exception: pass
        scores["SI_SDR"].append(si_sdr(c, y))
    means = {k: float(np.mean(v)) if len(v)>0 else float("nan") for k,v in scores.items()}
    print("[score]", {k: round(v,4) for k,v in means.items()})
