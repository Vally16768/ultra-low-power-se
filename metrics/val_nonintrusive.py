# val_nonintrusive.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run non-intrusive metrics (DNSMOS + NISQA) over a dir of enhanced wavs.
Usage:
  python val_nonintrusive.py \
      --enhanced_dir runs/exp_mask_df_v2/val_wavs \
      --out_csv      runs/exp_mask_df_v2/nonintrusive_val.csv
"""
import os, argparse, glob, csv
import soundfile as sf

# Expect your wrappers to expose: dns_mos(wav, sr) -> dict with keys: SIG, BAK, OVR
# and nisqa_mos(wav, sr) -> float
from dnsmos import dns_mos
from nisqa import nisqa_mos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced_dir", required=True)
    ap.add_argument("--out_csv",      required=True)
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.enhanced_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"No wavs in: {args.enhanced_dir}")

    rows = [("file","dnsmos_sig","dnsmos_bak","dnsmos_ovr","nisqa_mos")]
    for w in wavs:
        audio, sr = sf.read(w)
        if audio.ndim > 1: audio = audio.mean(axis=1)

        dns = dns_mos(audio, sr)         # { 'SIG':..., 'BAK':..., 'OVR':... }
        mos = nisqa_mos(audio, sr)       # float

        rows.append((os.path.basename(w),
                     float(dns.get("SIG", 0.0)),
                     float(dns.get("BAK", 0.0)),
                     float(dns.get("OVR", 0.0)),
                     float(mos)))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"✔ Wrote {len(rows)-1} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
