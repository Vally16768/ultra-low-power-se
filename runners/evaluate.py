# runners/evaluate.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

from ._resolve import resolve_pathlike


def _exists(p: str | Path) -> bool:
    return Path(p).expanduser().exists()


def main(cfg: Dict[str, Any]) -> int:
    eval_cfg = cfg.get("eval", {})
    sr = int(eval_cfg.get("sr") or cfg.get("data", {}).get("sample_rate", 16000))
    metrics = eval_cfg.get("metrics", ["pesq", "stoi", "si_sdr", "seg_snr"])

    # ---- OFFLINE ----
    offline = eval_cfg.get("offline", {})
    manifest_off = offline.get("manifest") or cfg.get("data", {}).get("manifests", {}).get("test_offline") or "data/prepared/voicebank/test/manifests/pairs.csv"
    manifest_off = resolve_pathlike(manifest_off, cfg)
    outdir_off = resolve_pathlike(offline.get("outdir") or "artifacts/eval/mamba_unet/enhanced", cfg)
    os.makedirs(outdir_off, exist_ok=True)

    if not _exists(manifest_off):
        raise SystemExit(
            f"[eval:offline] Manifest lipsă: {manifest_off}\n→ Rulează `make datasets` sau setează cfg.eval.offline.manifest / data.manifests.test_offline."
        )

    print(f"[eval] Offline manifest: {manifest_off}")
    print(f"[eval] Offline outdir:   {outdir_off}")
    print(f"[eval] Metrics:          {metrics}  | SR={sr}")

    # 1) Enhance
    from runners.infer import enhance_dataset

    enhance_dataset(cfg, manifest_off, outdir_off, sr=sr)

    # 2) Score
    from runners.score import score_dir

    score_csv = Path(outdir_off).with_suffix(".scores.csv")
    score_dir(
        cfg=cfg,
        ref_manifest=manifest_off,
        est_dir=outdir_off,
        metrics=metrics,
        sr=sr,
        out_csv=str(score_csv),
    )

    # ---- STREAMING (opțional) ----
    streaming = eval_cfg.get("streaming", {})
    if bool(streaming.get("enabled", False)):
        manifest_stream = resolve_pathlike(
            streaming.get("manifest")
            or cfg.get("data", {}).get("manifests", {}).get("test_streaming")
            or "data/prepared/test_challenge/streaming/manifests/pairs.csv",
            cfg,
        )
        if not _exists(manifest_stream):
            print(f"[streaming] manifest absent: {manifest_stream} — sar peste testul de streaming.")
        else:
            print("[streaming] TODO: integrează pipeline-ul tău de streaming aici.")
    else:
        print("[eval] Streaming dezactivat (eval.streaming.enabled=false).")

    print(f"[eval] Gata. Scorurile offline sunt în: {score_csv}")
    return 0
