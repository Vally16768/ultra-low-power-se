#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNet1D Speech Enhancement: inference + optional evaluation (no fallbacks).

- Chunk/hop configurabile din CLI
- Ajustare automată la multiplu de downsampling al UNet (safe)
- sqrt-Hann overlap-add cu normalizare prin suma ferestrelor
- Opțiuni: --batch-size, --allow-mem-growth
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf

# --- Project imports (fără fallback) ------------------------------------------
from core.data.audio_utils import load_audio_mono
from core.data.resample_audio import resample_audio  # disponibil la nevoie

from metrics.metrics_logger import setup_metrics_logger
from metrics.pesq import pesq_score_safe
from metrics.stoi import stoi_score_safe
from metrics.snr import snr_noisy_safe, snr_enhanced_safe

# Init loggerul de metrici (implementarea ta)
setup_metrics_logger()

# --- Constante ----------------------------------------------------------------
SAMPLE_RATE: int = 16000
TARGET_DTYPE = np.float32
CLIP_TO_RANGE = True
HANN_GUARD: float = 1e-8

# Implicit, dar configurabile din CLI
DEFAULT_CHUNK_SECONDS: float = 6.0
# Hop-ul implicit va fi calculat ca 50% din chunk dacă nu este dat explicit
DEFAULT_BATCH_SIZE: int = 1


# --- Utilitare audio -----------------------------------------------------------
def _sanitize_wav(y: np.ndarray, clip: bool = True) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(TARGET_DTYPE, copy=False)
    if clip:
        y = np.clip(y, -1.0, 1.0)
    return y


def _hann_window(n: int) -> np.ndarray:
    if n <= 0:
        return np.ones(1, dtype=TARGET_DTYPE)
    return np.hanning(n).astype(TARGET_DTYPE, copy=False)


def _overlap_add(chunks: list[np.ndarray], hop: int, win: np.ndarray) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=TARGET_DTYPE)
    chunk_len = chunks[0].shape[0]
    out_len = hop * (len(chunks) - 1) + chunk_len
    acc = np.zeros(out_len, dtype=TARGET_DTYPE)
    wsum = np.zeros(out_len, dtype=TARGET_DTYPE)
    for i, c in enumerate(chunks):
        start = i * hop
        end = start + chunk_len
        acc[start:end] += c * win
        wsum[start:end] += win
    wsum = np.where(wsum <= HANN_GUARD, 1.0, wsum)
    return acc / wsum


def _chunk_audio(x: np.ndarray, chunk_len: int, hop: int) -> list[np.ndarray]:
    if chunk_len <= 0:
        return [x]
    if len(x) <= chunk_len:
        pad = chunk_len - len(x)
        if pad > 0:
            x = np.pad(x, (0, pad), mode="constant")
        return [x]
    chunks = []
    for start in range(0, len(x), hop):
        end = start + chunk_len
        c = x[start:end]
        if len(c) < chunk_len:
            c = np.pad(c, (0, chunk_len - len(c)), mode="constant")
        chunks.append(c)
        if end >= len(x):
            break
    return chunks


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


# --- Helpers UNet stride -------------------------------------------------------
def _next_multiple(n: int, m: int) -> int:
    if m <= 1:
        return n
    return ((n + m - 1) // m) * m


def _guess_unet_stride_multiple(model: tf.keras.Model, fallback: int = 16) -> int:
    """
    Estimează factorul total de downsampling (putere de 2) parcurgând straturile.
    Dacă nu se poate estima corect, revine cu un fallback (16 sau 32 sunt uzuale).
    """
    factor = 1
    try:
        for l in model.layers:
            cfg = getattr(l, 'get_config', lambda: {})()
            lname = getattr(l, 'name', '').lower()

            # Max/AveragePooling1D (pool_size)
            pool_size = None
            if 'pool' in lname:
                ps = cfg.get('pool_size', None)
                if ps is not None:
                    pool_size = ps[0] if isinstance(ps, (list, tuple)) else ps
            if pool_size:
                try:
                    factor *= int(pool_size)
                    continue
                except Exception:
                    pass

            # Conv1D cu strides > 1 și padding 'valid'
            if cfg.get('strides', None) is not None:
                st = cfg['strides']
                if isinstance(st, (list, tuple)):
                    st = st[0]
                if int(st) > 1 and cfg.get('padding', 'same').lower() == 'valid':
                    factor *= int(st)

            # Straturi UpSampling/Transposed conv NU se iau în calcul (doar down)
    except Exception:
        return fallback

    # Snap la cele mai comune puteri de 2
    for p in (2, 4, 8, 16, 32, 64, 128):
        if factor <= p:
            return p
    return fallback


# --- Enhancer -----------------------------------------------------------------
class Enhancer:
    def __init__(self, model_path: Path):
        print(f"[Load] Model from: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path), compile=False)

    def enhance(
        self,
        x_16k: np.ndarray,
        *,
        normalize: bool = False,
        no_chunk: bool = False,
        chunk_len: int | None = None,
        hop: int | None = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        x_16k: mono, float32, sanitizat la 16 kHz (folosește load_audio_mono înainte).
        """
        x = x_16k.reshape(-1)

        # Normalizare max-abs (dacă a fost folosită și la antrenare)
        if normalize:
            max_abs = float(np.max(np.abs(x)) + 1e-8)
            print(f"[Norm] Max-abs normalization activ (max={max_abs:.6f})")
        else:
            max_abs = 1.0
        x_in = x / max_abs

        if no_chunk:
            xin = x_in.reshape(1, -1, 1)
            y = self.model.predict(xin, batch_size=batch_size, verbose=0)
            y = np.squeeze(y, axis=(0, -1))
            y = _sanitize_wav(y, clip=CLIP_TO_RANGE)
            y_full = y
        else:
            assert chunk_len is not None and hop is not None, "chunk_len și hop sunt necesare pentru chunking."
            # sqrt-Hann pentru OLA de tip analysis/synthesis
            win = np.sqrt(_hann_window(chunk_len))
            chunks = _chunk_audio(x_in, chunk_len, hop)
            out_chunks = []
            for c in chunks:
                xin = c.reshape(1, -1, 1)
                y = self.model.predict(xin, batch_size=batch_size, verbose=0)
                y = np.squeeze(y, axis=(0, -1))
                y = _sanitize_wav(y, clip=CLIP_TO_RANGE)
                out_chunks.append(y)
            y_full = _overlap_add(out_chunks, hop, win)
            y_full = y_full[: len(x_in)]

        # Rescalare înapoi dacă s-a normalizat
        y_full = (y_full * max_abs).astype(TARGET_DTYPE, copy=False)
        if CLIP_TO_RANGE:
            y_full = np.clip(y_full, -1.0, 1.0)
        return y_full


# --- CSV helpers --------------------------------------------------------------
def _pairs_from_csv(csv_path: Path) -> list[tuple[Path, Path]]:
    pairs = []
    with csv_path.open("r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header and ("noisy" in header and "clean" in header):
            noisy_idx = header.index("noisy")
            clean_idx = header.index("clean")
            for row in rdr:
                if not row:
                    continue
                pairs.append((Path(row[noisy_idx].strip()), Path(row[clean_idx].strip())))
        else:
            if header and len(header) >= 2:
                pairs.append((Path(header[0].strip()), Path(header[1].strip())))
            for row in rdr:
                if not row:
                    continue
                pairs.append((Path(row[0].strip()), Path(row[1].strip())))
    return pairs


def _collect_wavs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.wav")])


# --- Main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="UNet1D Speech Enhancement — Inference & Eval (no fallbacks)")
    ap.add_argument("--model", required=True, help="Path la .keras sau SavedModel dir")
    ap.add_argument("--input", required=True, help="Fișier .wav zgomotos sau director")
    ap.add_argument("--output", required=True, help="Fișier .wav ieșire sau director")
    ap.add_argument("--eval-clean", dest="eval_clean", help="Referință curată .wav (mod single-file)")
    ap.add_argument("--pairs-csv", dest="pairs_csv", help="CSV cu coloane 'noisy,clean' (mod director)")
    ap.add_argument("--normalize", action="store_true", help="Activează normalizarea max-abs (ca la training)")
    ap.add_argument("--no-chunk", action="store_true", help="Dezactivează chunking (inferință one-shot)")

    ap.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                    help="Lungimea chunk-ului (secunde) dat la model (potrivită cu training).")
    ap.add_argument("--hop-seconds", type=float, default=None,
                    help="Pasul dintre chunk-uri (secunde). Implicit: chunk/2.")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="Batch size pentru model.predict (implicit 1).")
    ap.add_argument("--allow-mem-growth", action="store_true",
                    help="Activează TF GPU memory growth (utile pe stații multi-GPU).")

    args = ap.parse_args()

    # Opțional: memory growth pentru GPU
    if args.allow_mem_growth:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    model_path = Path(args.model)
    in_path = Path(args.input)
    out_path = Path(args.output)

    enhancer = Enhancer(model_path)

    # Estimează multiplu-ul de downsampling al UNet și aliniază chunk_len la acesta
    stride_mult = _guess_unet_stride_multiple(enhancer.model, fallback=16)

    # calcule în eșantioane
    chunk_len_samples = int(round(args.chunk_seconds * SAMPLE_RATE))
    chunk_len_samples = _next_multiple(chunk_len_samples, stride_mult)

    if args.hop_seconds is None:
        hop_samples = max(1, chunk_len_samples // 2)  # implicit 50%
    else:
        hop_samples = int(round(args.hop_seconds * SAMPLE_RATE))
        hop_samples = max(1, min(hop_samples, chunk_len_samples))

    print(f"[Config] sr={SAMPLE_RATE}Hz, chunk={chunk_len_samples} samp ({chunk_len_samples/SAMPLE_RATE:.3f}s), "
          f"hop={hop_samples} samp ({hop_samples/SAMPLE_RATE:.3f}s), stride_mult≈{stride_mult}, "
          f"batch_size={args.batch_size}, no_chunk={args.no_chunk}")

    # --- Single file mode
    if in_path.is_file():
        noisy_16k = load_audio_mono(str(in_path), SAMPLE_RATE)
        enh = enhancer.enhance(
            noisy_16k,
            normalize=args.normalize,
            no_chunk=args.no_chunk,
            chunk_len=chunk_len_samples,
            hop=hop_samples,
            batch_size=args.batch_size,
        )

        y_out_path = out_path if out_path.suffix.lower() == ".wav" else out_path.with_suffix(".wav")
        _write_wav(y_out_path, enh, SAMPLE_RATE)
        print(f"[Save] {y_out_path}")

        # Optional: evaluare pe perechea corectă
        if args.eval_clean:
            clean_16k = load_audio_mono(str(Path(args.eval_clean)), SAMPLE_RATE)

            # Aliniere lungimi
            L = min(len(clean_16k), len(enh), len(noisy_16k))
            clean = clean_16k[:L]
            enh_ = enh[:L]
            noisy_ref = noisy_16k[:L]

            snr_noisy = snr_noisy_safe(clean, noisy_ref)
            snr_enh = snr_enhanced_safe(clean, enh_)
            delta_snr = snr_enh - snr_noisy
            pesq_val = pesq_score_safe(clean, enh_, SAMPLE_RATE)
            stoi_val = stoi_score_safe(clean, enh_, SAMPLE_RATE, extended=False, auto_resample=True)

            print("[Eval] Results:")
            print(f"  SNR(clean,noisy): {snr_noisy:.2f} dB")
            print(f"  SNR(clean,enh):   {snr_enh:.2f} dB  (Δ = {delta_snr:+.2f} dB)")
            print(f"  PESQ:            {pesq_val:.3f}")
            print(f"  STOI:            {stoi_val:.3f}")
        return

    # --- Folder mode
    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        wavs = _collect_wavs(in_path)
        print(f"[Info] Found {len(wavs)} wav(s) under {in_path}")

        # Optional: evaluare pe perechi din CSV
        pairs = {}
        if args.pairs_csv:
            for noisy_p, clean_p in _pairs_from_csv(Path(args.pairs_csv)):
                pairs[Path(noisy_p).resolve()] = Path(clean_p).resolve()

        agg = {"N": 0, "sum_dsnr": 0.0, "sum_pesq": 0.0, "sum_stoi": 0.0,
               "cnt_pesq": 0, "cnt_stoi": 0}

        for wp in wavs:
            rel = wp.relative_to(in_path)
            out_file = (out_path / rel).with_suffix(".wav")

            x_16k = load_audio_mono(str(wp), SAMPLE_RATE)
            y_16k = enhancer.enhance(
                x_16k,
                normalize=args.normalize,
                no_chunk=args.no_chunk,
                chunk_len=chunk_len_samples,
                hop=hop_samples,
                batch_size=args.batch_size,
            )

            out_file.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(out_file, y_16k, SAMPLE_RATE)
            print(f"[Save] {out_file}")

            # Per-file eval dacă există în CSV
            if pairs and wp.resolve() in pairs:
                clean_p = pairs[wp.resolve()]
                c_16k = load_audio_mono(str(clean_p), SAMPLE_RATE)

                L = min(len(c_16k), len(y_16k), len(x_16k))
                c = c_16k[:L]
                y2 = y_16k[:L]
                x2 = x_16k[:L]

                snr_n = snr_noisy_safe(c, x2)
                snr_e = snr_enhanced_safe(c, y2)
                dsnr = snr_e - snr_n
                agg["sum_dsnr"] += dsnr
                agg["N"] += 1

                pv = pesq_score_safe(c, y2, SAMPLE_RATE)
                agg["sum_pesq"] += pv
                agg["cnt_pesq"] += 1
                sv = stoi_score_safe(c, y2, SAMPLE_RATE, extended=False, auto_resample=True)
                agg["sum_stoi"] += sv
                agg["cnt_stoi"] += 1

        if agg["N"] > 0:
            print("\n[Eval] Aggregate metrics (paired files):")
            print(f"  ΔSNR (mean): {agg['sum_dsnr']/agg['N']:+.2f} dB  (N={agg['N']})")
            if agg["cnt_pesq"] > 0:
                print(f"  PESQ (mean): {agg['sum_pesq']/agg['cnt_pesq']:.3f}  (N={agg['cnt_pesq']})")
            if agg["cnt_stoi"] > 0:
                print(f"  STOI (mean): {agg['sum_stoi']/agg['cnt_stoi']:.3f}  (N={agg['cnt_stoi']})")
        return

    print(f"[Error] Input path not found: {in_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
