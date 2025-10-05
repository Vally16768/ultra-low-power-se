#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
import soundfile as sf


def read_list(p):
    return [Path(x.strip()) for x in open(p) if x.strip()]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x) + 1e-12)))


def load_mono_16k(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"SR!=16k: {p}")
    return x


def mix_one(cfile, nfile, snr_db):
    s = load_mono_16k(cfile)
    n = load_mono_16k(nfile)
    if len(n) < len(s):
        n = np.tile(n, int(np.ceil(len(s) / len(n))))[: len(s)]
    else:
        i = np.random.randint(0, len(n) - len(s) + 1)
        n = n[i : i + len(s)]
    tgt = rms(s) / (10 ** (snr_db / 20.0))
    b = rms(n) or 1e-6
    n = n * (tgt / b)
    y = s + n
    peak = max(np.max(np.abs(y)), 1.0)
    if peak > 0.99:
        y = y / peak * 0.99
    return y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--noise-list", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pairs", type=int, default=10000)
    ap.add_argument("--snr", default="0,5,10,15")
    a = ap.parse_args()
    out = Path(a.out_root)
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    cleans, noises = read_list(a.clean_list), read_list(a.noise_list)
    snrs = [float(x) for x in a.snr.split(",")]
    rng = np.random.default_rng(0)
    for i in range(a.pairs):
        c = cleans[int(rng.integers(0, len(cleans)))]
        n = noises[int(rng.integers(0, len(noises)))]
        s = snrs[int(rng.integers(0, len(snrs)))]
        y, sig = mix_one(c, n, s)
        stem = f"mix_{i:07d}"
        sf.write(str(out / "noisy" / f"{stem}.wav"), y, 16000)
        sf.write(str(out / "clean" / f"{stem}.wav"), sig, 16000)
        if (i + 1) % 1000 == 0:
            print(f"[mix] {i+1}/{a.pairs}")
    print(f"[done] wrote to {out}")


if __name__ == "__main__":
    main()
