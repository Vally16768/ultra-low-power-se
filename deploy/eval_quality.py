# eval_quality.py
#!/usr/bin/env python3
import argparse, numpy as np, soundfile as sf, librosa, json
from pathlib import Path

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

def load_wav(p, target_sr):
    x, sr = sf.read(p)
    if x.ndim == 2: x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return x, sr

def main():
    ap = argparse.ArgumentParser("Evaluate enhanced WAVs vs references")
    ap.add_argument("--clean_dir", required=True, help="Directory with clean refs (matching names)")
    ap.add_argument("--noisy_dir", required=True, help="Directory with noisy inputs (for SNRi)")
    ap.add_argument("--enh_dir",   required=True, help="Directory with enhanced outputs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out_json", default="artifacts/onnx_eval/metrics.json")
    args = ap.parse_args()

    clean_files = {p.name: p for p in Path(args.clean_dir).rglob("*.wav")}
    enh_files   = {p.name: p for p in Path(args.enh_dir).rglob("*.wav")}
    noisy_files = {p.name: p for p in Path(args.noisy_dir).rglob("*.wav")}
    names = sorted(set(clean_files) & set(enh_files) & set(noisy_files))
    assert names, "No matching filenames across clean/noisy/enhanced."

    results = []
    for nm in names:
        ref, _ = load_wav(clean_files[nm], args.sr)
        deg, _ = load_wav(enh_files[nm],   args.sr)
        noz, _ = load_wav(noisy_files[nm], args.sr)

        L = min(len(ref), len(deg), len(noz))
        ref, deg, noz = ref[:L], deg[:L], noz[:L]

        sisnr_e = si_snr(ref, deg)
        sisnr_n = si_snr(ref, noz)
        sisnri  = sisnr_e - sisnr_n

        snr_e = snr(ref, deg)
        snr_n = snr(ref, noz)
        snri  = snr_e - snr_n

        pesq = try_pesq(args.sr, ref, deg)
        stoi = try_stoi(args.sr, ref, deg)

        results.append({
            "file": nm,
            "SI-SNR_clean_enh": float(sisnr_e),
            "SI-SNR_clean_noisy": float(sisnr_n),
            "SI-SNRi": float(sisnri),
            "SNR_clean_enh": float(snr_e),
            "SNR_clean_noisy": float(snr_n),
            "SNRi": float(snri),
            "PESQ": None if pesq is None else float(pesq),
            "STOI": None if stoi is None else float(stoi),
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"count": len(results), "items": results}, f, indent=2)
    print(f"[OK] Wrote {out_path}  (files={len(results)})")

if __name__ == "__main__":
    main()