"""
Oracolul: calculează metrici pe manifest (noisy, clean, [enhanced]),
verifică pragurile, iese cu rc!=0 dacă oricare prag e încălcat.
"""

import csv, json, sys, numpy as np, soundfile as sf
from .pesq_metric import pesq_score
from .stoi_metric import stoi_score
from .snr_metrics import delta_snr
from .sisdr import sisdr

DEFAULT_THRESH = {
    "PESQ": 3.00,
    "STOI": 0.93,
    "DELTA_SNR": 9.0,  # dB
    "SI_SDR": 10.0,  # dB, opțional
    "STREAM_LAT_MS": 40.0,  # doar pentru streaming runs (pass-through aici)
}


def _load(path):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(-1)
    return x, fs


def eval_pair(clean_path, noisy_path, enhanced_path):
    clean, fs = _load(clean_path)
    noisy, _ = _load(noisy_path)
    enh, _ = _load(enhanced_path)
    # metrici intruzive
    pesq = pesq_score(clean, enh, fs)
    stoi = stoi_score(clean, enh, fs)
    dsnr = delta_snr(clean, noisy, enh)
    sdr = sisdr(clean, enh)
    return {"PESQ": pesq, "STOI": stoi, "DELTA_SNR": dsnr, "SI_SDR": sdr}


def main(manifest_csv: str, out_json: str, thresholds=None):
    thr = {**DEFAULT_THRESH, **(thresholds or {})}
    rows, scores = [], []
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            sc = eval_pair(r["clean"], r["noisy"], r["enhanced"])
            scores.append(sc)
            rows.append({**r, **sc})

    # agregare
    import statistics as st

    agg = {k: st.mean([s[k] for s in scores]) for k in scores[0].keys()}

    # verdict
    failures = []
    for k, v in agg.items():
        if k in thr and v < thr[k]:
            failures.append((k, v, thr[k]))

    report = {"thresholds": thr, "aggregate": agg, "count": len(scores), "failures": failures, "rows": rows}
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # printf scurt
    print("[ORACLE]", {k: round(v, 4) for k, v in agg.items()})
    if failures:
        print("[ORACLE][FAIL]", failures)
        sys.exit(2)


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV cu coloane: noisy,clean,enhanced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None, help="JSON string sau path la .json")
    args = ap.parse_args()
    thr = None
    if args.thresholds:
        if args.thresholds.strip().startswith("{"):
            thr = json.loads(args.thresholds)
        else:
            with open(args.thresholds) as f:
                thr = json.load(f)
    main(args.manifest, args.out, thr)
