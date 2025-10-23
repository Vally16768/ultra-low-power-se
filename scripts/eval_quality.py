#!/usr/bin/env python3
import argparse, numpy as np, soundfile as sf, librosa, json
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

def si_snr(ref, est, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    s = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    e = est - s
    return 10*np.log10((np.dot(s, s) + eps) / (np.dot(e, e) + eps))

def snr(ref, est, eps=1e-8):
    e = ref - est
    return 10*np.log10((np.sum(ref**2)+eps)/(np.sum(e**2)+eps))

def try_pesq(sr, ref, deg):
    try:
        from pesq import pesq
        return float(pesq(sr, ref, deg, 'wb'))
    except Exception:
        return None

def try_stoi(sr, ref, deg):
    try:
        from pystoi import stoi
        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return None

def load_audio_any(p, target_sr):
    try:
        x, sr = sf.read(p)
        if x.ndim == 2: x = x.mean(axis=1)
        x = x.astype(np.float32)
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return x, sr
    except Exception:
        x, sr = librosa.load(p, sr=target_sr, mono=True)
        x = x.astype(np.float32)
        return x, sr

def index_by_stem(root):
    m = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            m.setdefault(p.stem, p)
    return m

def main():
    ap = argparse.ArgumentParser("Evaluate enhanced audio vs references")
    ap.add_argument("--clean_dir", required=True, help="Directory with clean refs (matching stems)")
    ap.add_argument("--noisy_dir", required=True, help="Directory with noisy inputs (for SNRi)")
    ap.add_argument("--enh_dir",   required=True, help="Directory with enhanced outputs (.wav)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out_json", default="artifacts/onnx_eval/metrics.json")
    args = ap.parse_args()

    clean = index_by_stem(args.clean_dir)
    noisy = index_by_stem(args.noisy_dir)
    enh   = index_by_stem(args.enh_dir)

    names = sorted(set(clean) & set(noisy) & set(enh))
    if not names:
        print(f"[SKIP] No matching stems across: {args.clean_dir}, {args.noisy_dir}, {args.enh_dir}")
        return

    results = []
    for nm in names:
        ref, _ = load_audio_any(clean[nm], args.sr)
        deg, _ = load_audio_any(enh[nm],   args.sr)
        noz, _ = load_audio_any(noisy[nm], args.sr)

        L = min(len(ref), len(deg), len(noz))
        ref, deg, noz = ref[:L], deg[:L], noz[:L]

        sisnr_e = si_snr(ref, deg); sisnr_n = si_snr(ref, noz); sisnri = sisnr_e - sisnr_n
        snr_e   = snr(ref, deg);     snr_n   = snr(ref, noz);   snri   = snr_e   - snr_n
        pesq = try_pesq(args.sr, ref, deg)
        stoi = try_stoi(args.sr, ref, deg)

        results.append({
            "stem": nm,
            "clean": str(clean[nm]),
            "noisy": str(noisy[nm]),
            "enh":   str(enh[nm]),
            "SI-SNRi": float(sisnri),
            "SNRi": float(snri),
            "PESQ": None if pesq is None else float(pesq),
            "STOI": None if stoi is None else float(stoi),
        })

    import statistics as st
    def mean(xs): xs = [v for v in xs if v is not None]; return None if not xs else float(st.mean(xs))
    summary = {
        "count": len(results),
        "avg_SI-SNRi": mean([r["SI-SNRi"] for r in results]),
        "avg_SNRi": mean([r["SNRi"] for r in results]),
        "avg_PESQ": mean([r["PESQ"] for r in results]),
        "avg_STOI": mean([r["STOI"] for r in results]),
    }

    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)

    print(f"[OK] Wrote {outp} (files={len(results)})")
    print("Summary:", summary)

if __name__ == "__main__":
    main()
