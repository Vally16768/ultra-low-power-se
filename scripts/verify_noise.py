#!/usr/bin/env python3
import sys, json, math
from pathlib import Path
import soundfile as sf
import numpy as np

"""
Verifică data/noise/**/*.wav:
 - sample rate (implicit 16k), canale = 1, durată > 1.0s
 - detectează fișiere corupte
 - raport sumar JSON și text
Exit code != 0 dacă se găsesc probleme.
"""

ROOT = Path("data/noise")
SR_EXPECT = 16000
MIN_DUR = 1.0

def iter_wavs(root: Path):
    return sorted(root.rglob("*.wav"))

def stats(files):
    ok, bad = [], []
    sr_cnt, ch_cnt = {}, {}
    total_dur = 0.0
    for p in files:
        try:
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            ch = x.shape[1]
            dur = x.shape[0] / float(sr) if sr > 0 else 0.0
            sr_cnt[sr] = sr_cnt.get(sr, 0) + 1
            ch_cnt[ch] = ch_cnt.get(ch, 0) + 1
            total_dur += dur
            probs = []
            if sr != SR_EXPECT: probs.append(f"sr={sr}")
            if ch != 1: probs.append(f"ch={ch}")
            if dur < MIN_DUR: probs.append(f"dur={dur:.2f}s")
            (bad if probs else ok).append((p, probs))
        except Exception as e:
            bad.append((p, [f"read_error:{e}"]))
    return ok, bad, sr_cnt, ch_cnt, total_dur

def main():
    files = iter_wavs(ROOT)
    if not files:
        print("[verify_noise] ERROR: no *.wav under data/noise")
        sys.exit(2)
    ok, bad, sr_cnt, ch_cnt, total_dur = stats(files)
    print(f"[verify_noise] files: {len(files)} | ok: {len(ok)} | bad: {len(bad)} | dur: {total_dur/3600:.2f} h")
    print(f"  SR dist : {sr_cnt}")
    print(f"  CH dist : {ch_cnt}")
    if bad:
        for p, probs in bad[:20]:
            print(f"  BAD: {p} -> {', '.join(probs)}")
        print(f"[verify_noise] FAIL: {len(bad)} problematic files.")
        sys.exit(3)
    print("[verify_noise] OK")
    # JSON raport (opțional, util pt. CI)
    rep = {"files": len(files), "ok": len(ok), "bad": len(bad),
           "sr_dist": sr_cnt, "ch_dist": ch_cnt, "hours": total_dur/3600.0}
    (Path("data")/"reports").mkdir(parents=True, exist_ok=True)
    (Path("data")/"reports"/"noise_report.json").write_text(json.dumps(rep, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
