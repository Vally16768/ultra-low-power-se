#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict, Any
import numpy as np, soundfile as sf

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
    outdir = Path(cfg["eval"].get("outdir", "artifacts/eval/max/enhanced"))
    pairs_csv = cfg["data"]["test_csv"]
    ref_map = {}
    with open(pairs_csv,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_map[Path(row["noisy"]).stem] = row["clean"]

    scores = {"PESQ": [], "STOI": [], "SI_SDR": []}
    for wav in sorted(outdir.glob("*_enh.wav")):
        key = wav.stem.replace("_enh","")
        clean = ref_map.get(key)
        if not clean: 
            continue
        y,_ = sf.read(str(wav), dtype="float32"); c,_ = sf.read(clean, dtype="float32")
        L=min(len(y),len(c)); y=y[:L]; c=c[:L]
        if pesq is not None:
            try: scores["PESQ"].append(pesq(16000, c, y, "wb"))
            except Exception: pass
        if _stoi is not None:
            try: scores["STOI"].append(_stoi(c, y, 16000, extended=False))
            except Exception: pass
        scores["SI_SDR"].append(si_sdr(c, y))
    means = {k: float(np.mean(v)) if len(v)>0 else float("nan") for k,v in scores.items()}
    print("[score]", {k: round(v,4) for k,v in means.items()})
