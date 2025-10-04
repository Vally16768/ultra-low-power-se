# runners/score.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

from ._resolve import resolve_pathlike

try:
    from pesq import pesq
except Exception:
    pesq = None

try:
    from pystoi import stoi
except Exception:
    stoi = None


def _read_pairs(manifest: str | Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(manifest, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((row["noisy"], row["clean"]))
    return out


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _sdr_si(x_hat: np.ndarray, s: np.ndarray, eps=1e-8) -> float:
    s = s - np.mean(s)
    x_hat = x_hat - np.mean(x_hat)
    alpha = np.dot(x_hat, s) / (np.dot(s, s) + eps)
    e_target = alpha * s
    e_noise = x_hat - e_target
    ratio = (np.sum(e_target**2) + eps) / (np.sum(e_noise**2) + eps)
    return 10.0 * np.log10(ratio + eps)


def _seg_snr(x_hat: np.ndarray, s: np.ndarray, sr: int, frame_ms=20, eps=1e-8) -> float:
    N = int(sr * frame_ms / 1000.0)
    if len(s) < N or len(x_hat) < N:
        return 10.0 * np.log10((np.sum(s**2) + eps) / (np.sum((s - x_hat) ** 2) + eps) + eps)
    M = min(len(s), len(x_hat))
    s = s[:M]
    x_hat = x_hat[:M]
    num = 0.0
    den = 0.0
    k = 0
    for i in range(0, M, N):
        ss = s[i : i + N]
        xx = x_hat[i : i + N]
        if len(ss) < N or len(xx) < N:
            break
        num += np.sum(ss**2)
        den += np.sum((ss - xx) ** 2)
        k += 1
    if k == 0:
        return 0.0
    return 10.0 * np.log10((num + eps) / (den + eps) + eps)


def score_dir(cfg: Dict[str, Any], ref_manifest: str | Path, est_dir: str | Path, metrics: List[str], sr: int, out_csv: str | Path) -> None:
    pairs = _read_pairs(ref_manifest)
    est_dir = Path(est_dir)

    rows_out: List[Dict[str, float | str]] = []
    sums = {m: 0.0 for m in metrics}
    cnt_eff = {m: 0 for m in metrics}

    for noisy_path, clean_path in pairs:
        est_path = est_dir / Path(noisy_path).name
        if not est_path.exists():
            print(f"[score] Lipsă enhanced pentru {noisy_path} → {est_path.name}")
            continue

        ref, sr1 = sf.read(clean_path, always_2d=False)
        est, sr2 = sf.read(est_path, always_2d=False)
        ref = _mono(np.asarray(ref, dtype=np.float32))
        est = _mono(np.asarray(est, dtype=np.float32))
        if sr1 != sr or sr2 != sr:
            raise SystemExit(f"[score] SR neuniform (ref={sr1}, est={sr2}) — folosește {sr} Hz peste tot.")

        row = {"file": Path(noisy_path).name}
        for m in metrics:
            if m == "pesq":
                val = float("nan") if pesq is None else float(pesq(sr, ref, est, "wb"))
            elif m == "stoi":
                val = float("nan") if stoi is None else float(stoi(ref, est, sr, extended=False))
            elif m == "si_sdr":
                val = _sdr_si(est, ref)
            elif m == "seg_snr":
                val = _seg_snr(est, ref, sr)
            else:
                raise SystemExit(f"[score] Metrică necunoscută: {m}")
            row[m] = val
            if not np.isnan(val):
                sums[m] += val
                cnt_eff[m] += 1
        rows_out.append(row)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file"] + metrics
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("[score] Rezumat medii pe set (ignorat NaN):")
    for m in metrics:
        n_eff = cnt_eff[m]
        avg = (sums[m] / max(n_eff, 1)) if n_eff > 0 else float("nan")
        print(f"  - {m}: {avg:.4f}  (n={n_eff})")
    print(f"[score] Scris per-fișier: {out_csv}")


def main(cfg: Dict[str, Any]) -> int:
    eval_cfg = cfg.get("eval", {})
    sr = int(eval_cfg.get("sr") or cfg.get("data", {}).get("sample_rate", 16000))
    metrics = eval_cfg.get("metrics", ["pesq", "stoi", "si_sdr", "seg_snr"])

    offline = eval_cfg.get("offline", {})
    ref_manifest = offline.get("manifest") or cfg.get("data", {}).get("manifests", {}).get("test_offline")
    ref_manifest = resolve_pathlike(ref_manifest, cfg)
    if not ref_manifest:
        raise SystemExit("[score] Trebuie ref_manifest în cfg.eval.offline.manifest sau data.manifests.test_offline")

    est_dir = resolve_pathlike(offline.get("outdir") or "artifacts/eval/mamba_unet/enhanced", cfg)
    out_csv = Path(est_dir).with_suffix(".scores.csv")

    score_dir(cfg, ref_manifest, est_dir, metrics, sr, out_csv)
    return 0
