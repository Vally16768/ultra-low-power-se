from __future__ import annotations

import argparse
import json
from pathlib import Path

from classic_baselines.config import DSPConfig
from classic_baselines.evaluate import evaluate_manifest
from classic_baselines.io import ensure_dir, load_audio, read_manifest, save_audio
from classic_baselines.methods import get_method, list_methods
from classic_baselines.pca import DEFAULT_PCA_MODEL, fit_pca_model, load_pca_model

def run_method(
    method_name: str,
    test_csv: str | Path,
    out_dir: str | Path,
    cfg: DSPConfig,
    train_csv: str | Path | None = None,
    pca_model_path: str | Path = DEFAULT_PCA_MODEL,
) -> dict:
    method = get_method(method_name)
    out_dir = ensure_dir(out_dir)
    enh_dir = ensure_dir(Path(out_dir) / "enhanced")

    pca_model = None
    pca_path = Path(pca_model_path)
    if method.REQUIRES_PCA:
        if not pca_path.exists():
            if train_csv is None:
                raise FileNotFoundError(
                    f"PCA model not found: {pca_path}. Provide --train_csv or pre-fit the model with classic_baselines.pca."
                )
            fit_pca_model(train_csv, pca_path, cfg)
        pca_model = load_pca_model(pca_path)

    rows = read_manifest(test_csv)
    for row in rows:
        noisy_wav, _ = load_audio(row["noisy"], cfg.sr)
        enhanced = method.enhance(noisy_wav, cfg, pca_model=pca_model)
        save_audio(enh_dir / f"{row['stem']}.wav", enhanced, cfg.sr)

    config_path = Path(out_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "method": method_name,
                "test_csv": str(test_csv),
                "train_csv": None if train_csv is None else str(train_csv),
                "pca_model": None if pca_model is None else str(pca_path),
                "dsp": cfg.to_dict(),
            },
            f,
            indent=2,
        )

    return evaluate_manifest(
        test_csv=test_csv,
        enh_dir=enh_dir,
        out_json=Path(out_dir) / "metrics.json",
        out_csv=Path(out_dir) / "per_file.csv",
        sr=cfg.sr,
    )

def main() -> None:
    ap = argparse.ArgumentParser("Run one classic baseline on VoiceBank+DEMAND")
    ap.add_argument("--method", required=True, choices=list_methods())
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--pca_model", default=str(DEFAULT_PCA_MODEL))
    args = ap.parse_args()

    report = run_method(
        method_name=args.method,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        cfg=DSPConfig(),
        train_csv=args.train_csv,
        pca_model_path=args.pca_model,
    )
    print("[OK] Summary:", report["summary"])

if __name__ == "__main__":
    main()
