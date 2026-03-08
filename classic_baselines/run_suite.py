from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from classic_baselines.config import DSPConfig
from classic_baselines.io import ensure_dir
from classic_baselines.methods import list_methods
from classic_baselines.pca import DEFAULT_PCA_MODEL, fit_pca_model
from classic_baselines.run_method import run_method

def main() -> None:
    ap = argparse.ArgumentParser("Run the full classic-baseline suite on VoiceBank+DEMAND")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pca_model", default=str(DEFAULT_PCA_MODEL))
    args = ap.parse_args()

    cfg = DSPConfig()
    out_root = ensure_dir(args.out_root)
    pca_model = Path(args.pca_model)
    if not pca_model.exists():
        fit_pca_model(args.train_csv, pca_model, cfg)

    leaderboard = []
    for method in list_methods():
        report = run_method(
            method_name=method,
            test_csv=args.test_csv,
            out_dir=Path(out_root) / method,
            cfg=cfg,
            train_csv=args.train_csv,
            pca_model_path=pca_model,
        )
        row = {"method": method}
        row.update(report["summary"])
        leaderboard.append(row)

    json_path = Path(out_root) / "leaderboard.json"
    csv_path = Path(out_root) / "leaderboard.csv"
    with open(json_path, "w") as f:
        json.dump({"items": leaderboard}, f, indent=2)
    fieldnames = ["method"] + sorted({key for row in leaderboard for key in row.keys() if key != "method"})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print("[OK] Leaderboard written:", json_path)
    print("[OK] Leaderboard CSV:", csv_path)

if __name__ == "__main__":
    main()
