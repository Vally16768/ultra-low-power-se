# eval/aggregate_grid.py
import argparse, json, os, glob, csv
from pathlib import Path

FIELDS = ["variant", "model_path", "size_mb", "pesq_mean", "stoi_mean", "snr_impr_mean", "lat_mean_ms", "lat_p50_ms", "lat_p90_ms", "lat_p99_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for jf in sorted(glob.glob(os.path.join(args.dir, "*_metrics.json"))):
        with open(jf, "r") as f:
            data = json.load(f)
        model_path = data["model"]
        try:
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            size_mb = 0.0
        met = data.get("metrics", {})
        lat = data.get("latency_ms", {})
        rows.append(
            {
                "variant": Path(jf).stem.replace("_metrics", ""),
                "model_path": model_path,
                "size_mb": f"{size_mb:.2f}",
                "pesq_mean": f"{met.get('pesq', {}).get('mean', '')}",
                "stoi_mean": f"{met.get('stoi', {}).get('mean', '')}",
                "snr_impr_mean": f"{met.get('snr', {}).get('mean', '')}",
                "lat_mean_ms": f"{lat.get('mean', '')}",
                "lat_p50_ms": f"{lat.get('p50', '')}",
                "lat_p90_ms": f"{lat.get('p90', '')}",
                "lat_p99_ms": f"{lat.get('p99', '')}",
            }
        )

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {args.out} ({len(rows)} intrări)")


if __name__ == "__main__":
    main()
