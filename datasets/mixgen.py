from __future__ import annotations
import argparse, json, os, random, math, subprocess, uuid, csv
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import soundfile as sf

def set_seed(seed:int):
    import numpy as _np, random as _rnd, os as _os
    _os.environ["PYTHONHASHSEED"]=str(seed); _rnd.seed(seed); _np.random.seed(seed)

def rms(x): return np.sqrt(np.mean(np.maximum(1e-12, x**2)))
def norm_len(x, n):
    if len(x) == n: return x
    if len(x) > n:  return x[:n]
    y = np.zeros(n, dtype=x.dtype); y[:len(x)] = x; return y

def load_wav(path, sr):
    x, s = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(-1)
    if s != sr:
        import resampy
        x = resampy.resample(x, s, sr)
    return x

def convolve_rir(x, rir):
    y = np.convolve(x, rir, mode="full")
    y = y[:len(x)]
    g = rms(y) / (rms(x)+1e-12)
    return y / (g+1e-12)

def apply_clipping(x, mode="hard", thr=0.95):
    if mode=="hard": return np.clip(x, -thr, thr)
    return np.tanh(x / thr)

def apply_opus_codec(x, sr, kbps=16):
    tid = uuid.uuid4().hex
    tmp_in  = f"/tmp/{tid}_in.wav"
    tmp_out = f"/tmp/{tid}_out.wav"
    sf.write(tmp_in, x, sr)
    subprocess.run([
        "ffmpeg","-y","-loglevel","error",
        "-i", tmp_in, "-c:a","libopus","-b:a", f"{kbps}k", tmp_out
    ], check=True)
    y, s = sf.read(tmp_out, dtype="float32")
    os.remove(tmp_in); os.remove(tmp_out)
    if s != sr:
        import resampy
        y = resampy.resample(y, s, sr)
    if y.ndim>1: y=y.mean(-1)
    return y

@dataclass
class MixMeta:
    id: str
    clean_path: str
    noise_path: str
    rir_path: str | None
    snr_db: float
    sr: int
    codec: str | None
    codec_kbps: int | None
    clipping: str | None
    target_len: int
    offset_clean: int
    offset_noise: int
    gain_noise_db: float
    peak_before: float
    peak_after: float

def mix_one(clean, noise, sr, snr_db, rir_path=None, codec=None, codec_kbps=None, clip_mode=None, target_len=None, seed=0):
    rng = random.Random(seed)
    c = load_wav(clean, sr)
    n = load_wav(noise, sr)

    if target_len is None:
        target_len = rng.randint(int(2*sr), int(8*sr))
    if len(c) < target_len: c = norm_len(c, target_len)
    if len(n) < target_len: n = np.pad(n, (0, target_len-len(n)))
    off_c = 0
    off_n = rng.randint(0, max(0, len(n)-target_len))
    c = c[off_c:off_c+target_len]
    n = n[off_n:off_n+target_len]

    if rir_path:
        rir = load_wav(rir_path, sr)
        c = convolve_rir(c, rir)

    c_rms = rms(c); n_rms = rms(n)
    if n_rms < 1e-8:
        n = np.random.randn(*n.shape).astype(np.float32) * 1e-6
        n_rms = rms(n)
    target_noise_rms = c_rms / (10**(snr_db/20.0))
    noise_gain = target_noise_rms / (n_rms + 1e-12)
    y = c + n * noise_gain

    if codec in ("opus_16","opus_24"):
        kbps = 16 if codec=="opus_16" else 24
        y = apply_opus_codec(y, sr, kbps)
        y = norm_len(y, target_len)

    peak_before = float(np.max(np.abs(y)))
    if clip_mode:
        y = apply_clipping(y, mode=clip_mode, thr=0.95)
    peak_after = float(np.max(np.abs(y)))

    meta = MixMeta(
        id=uuid.uuid4().hex,
        clean_path=str(clean), noise_path=str(noise),
        rir_path=str(rir_path) if rir_path else None,
        snr_db=float(snr_db), sr=sr,
        codec=codec, codec_kbps={"opus_16":16,"opus_24":24}.get(codec),
        clipping=clip_mode, target_len=int(target_len),
        offset_clean=int(off_c), offset_noise=int(off_n),
        gain_noise_db=20*math.log10(noise_gain+1e-12),
        peak_before=peak_before, peak_after=peak_after
    )
    return y.astype(np.float32), c.astype(np.float32), meta

def verify_out_dir(out_dir: Path, sr_expected: int) -> tuple[bool, str]:
    man_path = out_dir / "manifests" / "pairs.csv"
    if not man_path.exists():
        return False, f"missing manifest: {man_path}"
    total = 0
    with open(man_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            total += 1
            noisy = row["noisy"].strip()
            clean = row["clean"].strip()
            meta  = row["meta"].strip() if "meta" in row else ""
            for p in (noisy, clean):
                if not Path(p).exists():
                    return False, f"missing file: {p}"
                try:
                    _, sr = sf.read(p, dtype="float32", always_2d=False)
                    if sr != sr_expected:
                        return False, f"bad SR {sr} for {p} (expected {sr_expected})"
                except Exception as e:
                    return False, f"cannot read {p}: {e}"
            if meta and not Path(meta).exists():
                return False, f"missing meta: {meta}"
    if total == 0:
        return False, "empty manifest"
    return True, f"verified {total} pairs"

def append_row(path: Path, noisy: Path, clean: Path, meta: Path | str):
    newfile = not path.exists()
    with open(path, "a", newline="") as f:
        if newfile:
            f.write("noisy,clean,meta\n")
        f.write(f"{noisy},{clean},{meta}\n")

def main():
    ap = argparse.ArgumentParser("mixgen")
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--noise-list", required=True)
    ap.add_argument("--rir-list", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snr", nargs="+", type=float, required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mixes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--reverb-prob", type=float, default=0.0)
    ap.add_argument("--codec", choices=["none","opus_16","opus_24"], default="none")
    ap.add_argument("--clipping", choices=["none","hard","soft"], default="none")
    ap.add_argument("--segment-min", type=float, default=2.0)
    ap.add_argument("--segment-max", type=float, default=8.0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir/"wav").mkdir(parents=True, exist_ok=True)
    (out_dir/"metadata").mkdir(parents=True, exist_ok=True)
    (out_dir/"manifests").mkdir(parents=True, exist_ok=True)
    man_path = out_dir/"manifests"/"pairs.csv"

    if args.verify_only:
        ok, msg = verify_out_dir(out_dir, args.sr)
        print(f"[mixgen][verify] {msg}")
        raise SystemExit(0 if ok else 1)

    if man_path.exists() and not args.force:
        ok, msg = verify_out_dir(out_dir, args.sr)
        if ok:
            print(f"[mixgen] dataset already OK → {msg} — nothing to do.")
            return
        else:
            print(f"[mixgen] manifest exists but invalid → {msg} — will (re)generate.")

    cleans = [l.strip() for l in open(args.clean_list) if l.strip()]
    noises = [l.strip() for l in open(args.noise_list) if l.strip()]
    rirs   = [l.strip() for l in open(args.rir_list)] if args.rir_list else []
    if not cleans or not noises:
        raise SystemExit("Empty clean/noise lists.")

    existing_ids = set()
    if man_path.exists():
        with open(man_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                noisy = Path(row["noisy"].strip()).name
                stem  = noisy.replace("_noisy.wav","")
                existing_ids.add(stem)

    wrote = 0
    for i in range(args.n_mixes):
        clean = cleans[i % len(cleans)]
        noise = noises[i % len(noises)]
        rir_p = random.choice(rirs) if (rirs and random.random() < args.reverb_prob) else None
        snr_db = random.choice(args.snr)
        seg_len = int(random.uniform(args.segment_min, args.segment_max) * args.sr)

        # id determinist (permite skip)
        uid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{args.seed}-{i}-{clean}-{noise}-{snr_db}-{seg_len}-{rir_p}-{args.codec}-{args.clipping}").hex
        if uid in existing_ids:
            continue

        y, c, m = mix_one(
            clean, noise, args.sr, snr_db, rir_p,
            codec=None if args.codec=="none" else args.codec,
            clip_mode=None if args.clipping=="none" else args.clipping,
            target_len=seg_len, seed=args.seed+i
        )
        m.id = uid

        noisy_path = out_dir/"wav"/f"{m.id}_noisy.wav"
        clean_path = out_dir/"wav"/f"{m.id}_clean.wav"
        meta_path  = out_dir/"metadata"/f"{m.id}.json"

        if not (noisy_path.exists() and clean_path.exists() and meta_path.exists()):
            sf.write(noisy_path, y, args.sr)
            sf.write(clean_path, c, args.sr)
            with open(meta_path,"w") as f: json.dump(asdict(m), f, indent=2)

        append_row(man_path, noisy_path, clean_path, meta_path)
        wrote += 1

    ok, msg = verify_out_dir(out_dir, args.sr)
    print(f"[mixgen] wrote {wrote} new pairs. Verify: {msg}")
    raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()
