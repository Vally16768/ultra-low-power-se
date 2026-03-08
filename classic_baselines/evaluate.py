from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from classic_baselines.io import ensure_dir, load_audio, read_manifest
from metrics.composite import evaluate_pair_metrics

SUMMARY_KEYS = [
    "PESQ", "CSIG", "CBAK", "COVL", "STOI", "SI_SDR", "SNR_IN", "SNR_OUT",
    "SNRi", "SI-SNRi", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVR",
]

def evaluate_manifest(
    test_csv: str | Path,
    enh_dir: str | Path,
    out_json: str | Path,
    out_csv: str | Path,
    sr: int,
) -> dict:
    rows = read_manifest(test_csv)
    enh_dir = Path(enh_dir)
    per_file = []
    for row in rows:
        enh_path = enh_dir / f"{row['stem']}.wav"
        if not enh_path.exists():
            raise FileNotFoundError(f"Missing enhanced file for stem '{row['stem']}': {enh_path}")
        clean_wav, _ = load_audio(row["clean"], sr)
        noisy_wav, _ = load_audio(row["noisy"], sr)
        enh_wav, _ = load_audio(enh_path, sr)
        item = {
            "stem": row["stem"],
            "clean": row["clean"],
            "noisy": row["noisy"],
            "enh": str(enh_path),
        }
        item.update(evaluate_pair_metrics(clean_wav, noisy_wav, enh_wav, sr, enhanced_path=str(enh_path)))
        per_file.append(item)

    summary = {"count": len(per_file)}
    for key in SUMMARY_KEYS:
        values = [float(item[key]) for item in per_file if key in item and item[key] is not None]
        summary[f"avg_{key}"] = None if not values else float(sum(values) / len(values))

    out_json = Path(out_json)
    out_csv = Path(out_csv)
    ensure_dir(out_json.parent)
    ensure_dir(out_csv.parent)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "items": per_file}, f, indent=2)
    fieldnames = ["stem", "clean", "noisy", "enh"] + SUMMARY_KEYS
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_file:
            writer.writerow({key: item.get(key) for key in fieldnames})
    return {"summary": summary, "items": per_file}

def main() -> None:
    ap = argparse.ArgumentParser("Evaluate classic baseline outputs against VoiceBank+DEMAND")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--enh_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    out_csv = args.out_csv or str(Path(args.out_json).with_name("per_file.csv"))
    report = evaluate_manifest(args.test_csv, args.enh_dir, args.out_json, out_csv, args.sr)
    print("[OK] Evaluated files:", report["summary"]["count"])
    print("[OK] Summary:", report["summary"])

if __name__ == "__main__":
    main()
