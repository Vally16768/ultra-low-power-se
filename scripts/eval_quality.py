#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.composite import evaluate_pair_metrics

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

def load_audio_any(p, target_sr):
    try:
        x, sr = sf.read(p)
        if x.ndim == 2: x = x.mean(axis=1)
        x = x.astype(np.float32)
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return x, sr
    except Exception:
        x, sr = librosa.load(p, sr=target_sr, mono=True)
        x = x.astype(np.float32)
        return x, sr

def index_by_stem(root):
    m = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            m.setdefault(p.stem, p)
    return m

def main():
    ap = argparse.ArgumentParser("Evaluate enhanced audio vs references")
    ap.add_argument("--clean_dir", required=True, help="Directory with clean refs (matching stems)")
    ap.add_argument("--noisy_dir", required=True, help="Directory with noisy inputs (for SNRi)")
    ap.add_argument("--enh_dir",   required=True, help="Directory with enhanced outputs (.wav)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out_json", default="artifacts/onnx_eval/metrics.json")
    ap.add_argument("--out_csv", default=None, help="Optional per-file CSV path")
    args = ap.parse_args()

    clean = index_by_stem(args.clean_dir)
    noisy = index_by_stem(args.noisy_dir)
    enh   = index_by_stem(args.enh_dir)

    names = sorted(set(clean) & set(noisy) & set(enh))
    if not names:
        print(f"[SKIP] No matching stems across: {args.clean_dir}, {args.noisy_dir}, {args.enh_dir}")
        return

    results = []
    for nm in names:
        ref, _ = load_audio_any(clean[nm], args.sr)
        deg, _ = load_audio_any(enh[nm],   args.sr)
        noz, _ = load_audio_any(noisy[nm], args.sr)

        row = {
            "stem": nm,
            "clean": str(clean[nm]),
            "noisy": str(noisy[nm]),
            "enh":   str(enh[nm]),
        }
        row.update(evaluate_pair_metrics(ref, noz, deg, args.sr, enhanced_path=str(enh[nm])))
        results.append(row)

    import statistics as st
    def mean(key):
        values = [float(r[key]) for r in results if key in r and r[key] is not None]
        return None if not values else float(st.mean(values))

    metric_keys = [
        "PESQ", "CSIG", "CBAK", "COVL", "STOI", "SI_SDR", "SNR_IN", "SNR_OUT",
        "SNRi", "SI-SNRi", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVR",
    ]
    summary = {"count": len(results)}
    for key in metric_keys:
        summary[f"avg_{key}"] = mean(key)

    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)

    if args.out_csv:
        import csv

        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["stem", "clean", "noisy", "enh"] + metric_keys
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"[OK] Wrote {outp} (files={len(results)})")
    print("Summary:", summary)

if __name__ == "__main__":
    main()
