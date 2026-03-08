"""
Evaluate intrusive metrics over a CSV manifest with columns:
   clean,noisy,enhanced
Writes a JSON report and exits non-zero if thresholds are violated.

Note: This script is for offline evaluation (not training).
"""
from __future__ import annotations
import csv, json, sys, os
import statistics as st
import numpy as np
import soundfile as sf
import librosa

from metrics.composite import evaluate_pair_metrics


DEFAULT_THRESH = {
    "PESQ": 3.00,
    "STOI": 0.93,
    "SNRi": 0.0,
    "SI_SDR": 10.0,     # dB
    "CSIG": 3.0,
    "CBAK": 3.0,
    "COVL": 3.0,
    "STREAM_LAT_MS": 40.0,  # placeholder for streaming runs
}

def _load(path):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)  # fold to mono
    return x.astype(np.float32), int(fs)

def _match_sr_to(ref_sig: np.ndarray, ref_fs: int, x: np.ndarray, x_fs: int):
    if x_fs != ref_fs:
        x = librosa.resample(x, orig_sr=x_fs, target_sr=ref_fs, res_type="kaiser_fast")
    n = min(len(ref_sig), len(x))
    return ref_sig[:n], x[:n], ref_fs

def eval_pair(clean_path, noisy_path, enhanced_path):
    clean, fs_c = _load(clean_path)
    noisy, fs_n = _load(noisy_path)
    enh, fs_e = _load(enhanced_path)

    # Normalize SR: align noisy/enhanced to clean's SR/length
    clean2, noisy2, fs_use = _match_sr_to(clean, fs_c, noisy, fs_n)
    clean3, enh2, _ = _match_sr_to(clean2, fs_use, enh, fs_e)

    return evaluate_pair_metrics(clean3, noisy2, enh2, fs_use)


def _read_manifest(manifest_csv: str):
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    required = {"clean", "noisy", "enhanced"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows


def main(manifest_csv: str, out_json: str, thresholds=None):
    thr = {**DEFAULT_THRESH, **(thresholds or {})}
    rows_in = _read_manifest(manifest_csv)

    scores, out_rows = [], []
    for r in rows_in:
        for k in ("clean", "noisy", "enhanced"):
            if not os.path.exists(r[k]):
                raise FileNotFoundError(f"{k} file not found: {r[k]}")
        sc = eval_pair(r["clean"], r["noisy"], r["enhanced"])
        scores.append(sc)
        out_rows.append({**r, **{k: float(v) for k, v in sc.items()}})

    agg = {k: st.mean([s[k] for s in scores]) for k in scores[0].keys()}
    failures = [(k, agg[k], thr[k]) for k in agg if k in thr and agg[k] < thr[k]]

    report = {
        "thresholds": thr,
        "aggregate": agg,
        "count": len(scores),
        "failures": failures,
        "rows": out_rows,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[ORACLE]", {k: round(v, 4) for k, v in agg.items()})
    if failures:
        print("[ORACLE][FAIL]", failures)
        sys.exit(2)


if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns: noisy,clean,enhanced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None, help="JSON string or path to .json")
    args = ap.parse_args()
    thr = None
    if args.thresholds:
        if args.thresholds.strip().startswith("{"):
            thr = _json.loads(args.thresholds)
        else:
            with open(args.thresholds) as f:
                thr = _json.load(f)
    main(args.manifest, args.out, thr)
