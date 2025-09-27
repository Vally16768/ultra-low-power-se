from __future__ import annotations
import argparse, json, os, random, math, subprocess, uuid
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
    # energy normalizare pentru a nu crește nivelul necontrolat
    g = rms(y) / (rms(x)+1e-12)
    return y / (g+1e-12)

def apply_clipping(x, mode="hard", thr=0.95):
    if mode=="hard":
        return np.clip(x, -thr, thr)
    # soft clipping (tanh)
    return np.tanh(x / thr)

def apply_opus_codec(x, sr, kbps=16):
    # scriem temp wav -> ffmpeg encode/decode opus → back to wav array
    tid = uuid.uuid4().hex
    tmp_in  = f"/tmp/{tid}_in.wav"
    tmp_out = f"/tmp/{tid}_out.wav"
    sf.write(tmp_in, x, sr)
    # use opus encoder via ffmpeg
    subprocess.run([
        "ffmpeg","-y","-loglevel","error",
        "-i", tmp_in,
        "-c:a","libopus","-b:a", f"{kbps}k",
        tmp_out
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

    # segmente random dacă sunt lungi
    if target_len is None:
        target_len = rng.randint(int(2*sr), int(8*sr))
    if len(c) < target_len: c = norm_len(c, target_len)
    if len(n) < target_len: n = np.pad(n, (0, target_len-len(n)))
    off_c = 0
    off_n = rng.randint(0, max(0, len(n)-target_len))
    c = c[off_c:off_c+target_len]
    n = n[off_n:off_n+target_len]

    # reverb pe clean (dacă e cazul)
    if rir_path:
        rir = load_wav(rir_path, sr)
        c = convolve_rir(c, rir)

    # potrivire SNR
    c_rms = rms(c); n_rms = rms(n)
    if n_rms < 1e-8:
        n = np.random.randn(*n.shape).astype(np.float32) * 1e-6
        n_rms = rms(n)
    target_noise_rms = c_rms / (10**(snr_db/20.0))
    noise_gain = target_noise_rms / (n_rms + 1e-12)
    y = c + n * noise_gain

    # codec degradare (după mix, ca în lumea reală transport)
    if codec in ("opus_16","opus_24"):
        kbps = 16 if codec=="opus_16" else 24
        y = apply_opus_codec(y, sr, kbps)
        # re-aliniere la lungime
        y = norm_len(y, target_len)

    # clipping (opțional)
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
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); (out_dir/"wav").mkdir(parents=True, exist_ok=True)
    (out_dir/"metadata").mkdir(parents=True, exist_ok=True)
    (out_dir/"manifests").mkdir(parents=True, exist_ok=True)

    cleans = [l.strip() for l in open(args.clean_list) if l.strip()]
    noises = [l.strip() for l in open(args.noise_list) if l.strip()]
    rirs   = [l.strip() for l in open(args.rir_list)] if args.rir_list else []

    meta_all = []
    for i in range(args.n_mixes):
        clean = cleans[i % len(cleans)]
        noise = noises[i % len(noises)]
        rir_p = None
        if rirs and random.random() < args.reverb_prob:
            rir_p = random.choice(rirs)
        snr_db = random.choice(args.snr)
        seg_len = int(random.uniform(args.segment_min, args.segment_max) * args.sr)
        y, c, m = mix_one(clean, noise, args.sr, snr_db, rir_p,
                          codec=None if args.codec=="none" else args.codec,
                          clip_mode=None if args.clipping=="none" else args.clipping,
                          target_len=seg_len, seed=args.seed+i)
        noisy_path = out_dir/"wav"/f"{m.id}_noisy.wav"
        clean_path = out_dir/"wav"/f"{m.id}_clean.wav"
        sf.write(noisy_path, y, args.sr); sf.write(clean_path, c, args.sr)
        meta_path  = out_dir/"metadata"/f"{m.id}.json"
        with open(meta_path,"w") as f: json.dump(asdict(m), f, indent=2)
        meta_all.append({"id":m.id, "noisy":str(noisy_path), "clean":str(clean_path), "meta":str(meta_path)})

    # manifest CSV (noisy,clean,meta)
    man_path = out_dir/"manifests"/"pairs.csv"
    with open(man_path,"w") as f:
        f.write("noisy,clean,meta\n")
        for r in meta_all:
            f.write(f"{r['noisy']},{r['clean']},{r['meta']}\n")
    print(f"[mixgen] wrote {len(meta_all)} pairs → {man_path}")

if __name__ == "__main__":
    main()
