#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/ensure_noise.sh [--demand /path/to/DEMAND] [--min-count 30] [--sr 16000]
#
# Creează structura data/noise/*, leagă DEMAND dacă e dat, iar dacă nu există
# niciun .wav -> generează zgomote sintetice (white/pink/brown).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMAND_ARG=""
MIN_COUNT=30
SR=16000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --demand)     DEMAND_ARG="$2"; shift 2 ;;
    --min-count)  MIN_COUNT="$2"; shift 2 ;;
    --sr)         SR="$2"; shift 2 ;;
    *) echo "[ensure_noise] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

NOISE_ROOT="data/noise"
DEMAND="$NOISE_ROOT/demand"
CUSTOM="$NOISE_ROOT/custom"
SYNTH="$NOISE_ROOT/synth"
DEV_DIR="$NOISE_ROOT/dev"
UNSEEN_DIR="$NOISE_ROOT/unseen"
mkdir -p "$DEMAND" "$CUSTOM" "$SYNTH" "$DEV_DIR" "$UNSEEN_DIR"

# leagă DEMAND ca symlink, dacă e furnizat și valid
if [[ -n "${DEMAND_ARG}" ]]; then
  if [[ -d "$DEMAND_ARG" ]]; then
    if [[ ! -e "$DEMAND" || -L "$DEMAND" ]]; then
      rm -f "$DEMAND"
      ln -s "$DEMAND_ARG" "$DEMAND"
      echo "[ensure_noise] linked DEMAND -> $DEMAND_ARG"
    else
      echo "[ensure_noise] DEMAND exists (skip link)"
    fi
  else
    echo "[ensure_noise] WARN: --demand '$DEMAND_ARG' nu e director; ignor."
  fi
fi

has_wav()      { find "$1" -type f -iname "*.wav" 2>/dev/null | head -n1 >/dev/null; }
count_wav_all(){ find "$NOISE_ROOT" -type f -iname "*.wav" 2>/dev/null | wc -l | tr -d ' '; }

# dacă nu există niciun .wav -> generează synth
if ! has_wav "$DEMAND" && ! has_wav "$CUSTOM" && ! has_wav "$SYNTH"; then
  echo "[ensure_noise] no noises found -> generating synthetic noises in $SYNTH ..."
  python - <<PY
import numpy as np, soundfile as sf
from pathlib import Path
SR=${SR}; OUT=Path("${SYNTH}"); OUT.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(1337)
def white(n): return (rng.standard_normal(n).astype(np.float32)*0.2)
def brown(n):
    w=rng.standard_normal(n).astype(np.float32)
    x=np.cumsum(w); x=x/(np.arange(n, dtype=np.float32)+1.0)
    x=x/(np.max(np.abs(x))+1e-9); return x*0.2
def pink(n):
    X=rng.standard_normal(n).astype(np.float32)
    F=np.fft.rfft(X); freqs=np.fft.rfftfreq(n,1.0/SR); F[1:]/=np.sqrt(freqs[1:]+1e-9)
    x=np.fft.irfft(F, n).astype(np.float32); x/= (np.max(np.abs(x))+1e-9); return x*0.2
N=SR*60
for i in range(10):
    sf.write(OUT/f"white_{i:02d}.wav", white(N), SR)
    sf.write(OUT/f"pink_{i:02d}.wav",  pink(N),  SR)
    sf.write(OUT/f"brown_{i:02d}.wav", brown(N), SR)
print("[ensure_noise] synth done ->", OUT)
PY
fi

TOTAL=$(count_wav_all)
echo "[ensure_noise] total .wav under data/noise: $TOTAL"
if [[ "$TOTAL" -lt "$MIN_COUNT" ]]; then
  echo "[ensure_noise] ERROR: prea puține fișiere de zgomot ($TOTAL < $MIN_COUNT)." >&2
  echo "  Adaugă DEMAND (--demand /path/DEMAND) sau pune .wav în data/noise/custom/." >&2
  exit 4
fi

echo "[ensure_noise] OK"
