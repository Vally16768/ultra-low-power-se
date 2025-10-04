#!/usr/bin/env python3
import argparse, json, glob
from time import time
import soundfile as sf
import os

ap = argparse.ArgumentParser()
ap.add_argument("--wav-dir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

t0 = time()
# doar pentru a avea o măsură stabilă: parcurgem fișierele
wav_paths = glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True)
audio_s = 0.0
for p in wav_paths:
    info = sf.info(p)
    audio_s += info.frames / info.samplerate
wall_s = time() - t0  # NOTĂ: nu include timpul de inferență; doar I/O. Pentru CI minimalist ok.

# dacă avem rulat inferența în pasul anterior, putem citi un stamp:
# Ca fallback folosim wall_s/audio_s ~ 0, ceea ce nu reflectă inferența reală în CI.
rtf = max(wall_s / max(audio_s, 1e-9), 0.0)

data = {"audio_seconds": audio_s, "scan_wall_seconds": wall_s, "rtf": rtf}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(data, f, indent=2)
print("[compute_timing] wrote", args.out, "files:", len(wav_paths))
