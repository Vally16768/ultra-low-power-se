#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.features.feature import FeatureConfig
from core.train import (
    load_feature_stats,
    load_manifest,
    reconstruct_from_logmel_strict,
    z_denorm_mel,
    z_norm_f0,
    z_norm_mel,
)
from metrics.composite import evaluate_pair_metrics


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _require_param(name, cli_value, fe_obj, fe_attr):
    if hasattr(fe_obj, fe_attr):
        return getattr(fe_obj, fe_attr)
    if cli_value is None:
        raise ValueError(
            f"FeatureConfig missing '{fe_attr}'. Provide '--{name.replace('_', '-')}'."
        )
    return cli_value


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
    return np.asarray(wav, dtype=np.float32)


def _build_normalized_input(
    npz_path: str,
    mel_mean: np.ndarray,
    mel_std: np.ndarray,
    f0_mean: float,
    f0_std: float,
) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    feats = data["feats"].astype(np.float32)
    if feats.shape[1] < 50:
        raise ValueError(f"Expected at least 50 feature dims in {npz_path}, got {feats.shape[1]}")

    mel_in = z_norm_mel(feats[:, :48], mel_mean, mel_std)
    f0_in = z_norm_f0(feats[:, 48], f0_mean, f0_std)[:, None]
    vprob = feats[:, 49:50]
    ceps = feats[:, 50:] if feats.shape[1] > 50 else None
    if ceps is None:
        return np.concatenate([mel_in, f0_in, vprob], axis=-1)
    return np.concatenate([mel_in, f0_in, vprob, ceps], axis=-1)


def main():
    ap = argparse.ArgumentParser("Evaluate a saved run with the current metric stack")
    ap.add_argument("--run_dir", required=True, type=Path, help="Run directory containing best_tf/")
    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--train_stats", required=True, type=Path)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--enh_dir", type=Path, default=None,
                    help="Optional directory for rendered enhanced wavs. Defaults to a temp dir.")
    ap.add_argument("--mel_ceps", type=int, default=0)
    ap.add_argument("--stft_center", type=_str2bool, default=None)
    ap.add_argument("--log_base", type=str, choices=["ln", "log10"], default=None)
    ap.add_argument("--log_eps", type=float, default=None)
    ap.add_argument("--peak_target", type=float, default=None)
    ap.add_argument("--cap_ratio", type=float, default=6.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--noisy_blend", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    model_path = run_dir / "best_tf"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {model_path}")

    out_json = args.out_json or (run_dir / "metrics_full.json")
    out_csv = args.out_csv or (run_dir / "per_file_full.csv")

    fcfg = FeatureConfig(mel_ceps_keep=args.mel_ceps)
    center = bool(_require_param("stft_center", args.stft_center, fcfg, "center"))
    log_base = str(_require_param("log_base", args.log_base, fcfg, "log_base"))
    log_eps = float(_require_param("log_eps", args.log_eps, fcfg, "log_eps"))
    peak_target = float(_require_param("peak_target", args.peak_target, fcfg, "peak_target"))

    mel_mean, mel_std, f0_mean, f0_std = load_feature_stats(args.train_stats)
    df_test = load_manifest(args.test_manifest)
    if args.limit is not None:
        df_test = df_test.iloc[: args.limit].copy()

    sr = int(fcfg.sr)
    stft_win = int(fcfg.win_length)
    stft_hop = int(fcfg.hop_length)
    n_fft_eff = int(max(int(fcfg.n_fft), int(fcfg.win_length)))
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft_eff,
        n_mels=48,
        fmin=float(fcfg.fmin),
        fmax=fcfg.fmax,
        htk=True,
    ).astype(np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)

    temp_ctx = None
    if args.enh_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"{run_dir.name}_full_eval_")
        enh_dir = Path(temp_ctx.name)
    else:
        enh_dir = args.enh_dir.resolve()
        enh_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"eval {run_dir.name}", ncols=100):
        npz_path = str(row["npz"])
        noisy_path = str(row["noisy"])
        clean_path = str(row["clean"])
        stem = Path(noisy_path).stem

        feats_n = _build_normalized_input(npz_path, mel_mean, mel_std, f0_mean, f0_std)
        inputs = feats_n[None, ...]
        time_mask = np.ones((1, feats_n.shape[0], 1), dtype=np.float32)
        pred_logmel_n = model.predict([inputs, time_mask], verbose=0)[0]
        pred_logmel = z_denorm_mel(pred_logmel_n, mel_mean, mel_std)

        noisy_wav = _load_audio(noisy_path, sr)
        clean_wav = _load_audio(clean_path, sr)
        enh_wav = reconstruct_from_logmel_strict(
            noisy_wav=noisy_wav,
            sr=sr,
            pred_logmel=pred_logmel,
            stft_win=stft_win,
            stft_hop=stft_hop,
            n_fft_eff=n_fft_eff,
            mel_fb=mel_fb,
            log_base=log_base,
            log_eps=log_eps,
            center=center,
            peak_target=peak_target,
            cap_ratio=args.cap_ratio,
            ridge=args.ridge,
            noisy_blend=args.noisy_blend,
        )

        enh_path = enh_dir / f"{stem}.wav"
        sf.write(str(enh_path), enh_wav, sr)

        item = {
            "stem": stem,
            "clean": clean_path,
            "noisy": noisy_path,
            "enh": str(enh_path),
        }
        item.update(
            evaluate_pair_metrics(
                clean=clean_wav,
                noisy=noisy_wav,
                enhanced=enh_wav,
                sr=sr,
                enhanced_path=str(enh_path),
            )
        )
        for key, value in item.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(f"{key} is not finite for {stem}: {value}")
        results.append(item)

    metric_keys = sorted(
        key for key in results[0].keys()
        if key not in {"stem", "clean", "noisy", "enh"}
    )
    summary = {"count": len(results)}
    for key in metric_keys:
        summary[f"avg_{key}"] = float(np.mean([float(row[key]) for row in results]))

    payload = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "summary": summary,
        "items": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "clean", "noisy", "enh"] + metric_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    if temp_ctx is not None:
        temp_ctx.cleanup()

    print(f"[OK] Wrote {out_json} (files={len(results)})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
