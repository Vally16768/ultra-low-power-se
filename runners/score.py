from __future__ import annotations
import os, sys, json, csv
from pathlib import Path
from typing import Dict, Any
import statistics as st
import soundfile as sf
import numpy as np
from metrics.pesq_metric import pesq_score
from metrics.stoi_metric import stoi_score
from metrics.snr_metrics import delta_snr
from metrics.sisdr import sisdr

def _load(path):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim>1: x=x.mean(-1)
    return x, fs

def score_manifest(manifest_csv: str) -> Dict[str, float]:
    rows = []
    with open(manifest_csv) as f:
        rd = csv.DictReader(f)
        for r in rd:
            c, fs = _load(r["clean"])
            n, _  = _load(r["noisy"])
            e, _  = _load(r["enhanced"])
            row = {
                "PESQ": pesq_score(c, e, fs),
                "STOI": stoi_score(c, e, fs),
                "DELTA_SNR": delta_snr(c, n, e),
                "SI_SDR": sisdr(c, e),
                "clean": r["clean"], "noisy": r["noisy"], "enhanced": r["enhanced"]
            }
            rows.append(row)
    agg = {k: st.mean([r[k] for r in rows]) for k in ["PESQ","STOI","DELTA_SNR","SI_SDR"]}
    return {"aggregate": agg, "rows": rows}

def _write_outputs(out_dir: str | Path, result: Dict[str, Any]):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out_dir/"metrics.csv","w") as f:
        w = csv.writer(f); w.writerow(["PESQ","STOI","DELTA_SNR","SI_SDR","clean","noisy","enhanced"])
        for r in result["rows"]:
            w.writerow([r["PESQ"], r["STOI"], r["DELTA_SNR"], r["SI_SDR"], r["clean"], r["noisy"], r["enhanced"]])
    print(f"[score] wrote {out_dir}/metrics.json & metrics.csv")

def _check_thresholds(agg: Dict[str,float], thr: Dict[str, float]) -> int:
    failures = []
    for k, t in thr.items():
        if k in agg and agg[k] < t:
            failures.append((k, agg[k], t))
    if failures:
        print("[score][FAIL]", failures)
        return 2
    print("[score][PASS]", {k: round(v,4) for k,v in agg.items()})
    return 0

def main(cfg: Dict[str, Any]):
    thresholds = cfg.get("evaluation",{}).get("thresholds", {"PESQ":3.0,"STOI":0.93,"DELTA_SNR":9.0,"SI_SDR":10.0})

    # scor standard
    man_std = "artifacts/infer/test_standard/pairs_eval.csv"
    res_std = score_manifest(man_std)
    _write_outputs("artifacts/score/test_standard", res_std)
    rc1 = _check_thresholds(res_std["aggregate"], thresholds)

    # scor streaming (dacă există)
    man_stream = Path("artifacts/infer/challenge_streaming/pairs_eval.csv")
    rc2 = 0
    if man_stream.exists():
        res_stream = score_manifest(str(man_stream))
        _write_outputs("artifacts/score/challenge_streaming", res_stream)
        rc2 = _check_thresholds(res_stream["aggregate"], thresholds)

    sys.exit( rc1 or rc2 )
