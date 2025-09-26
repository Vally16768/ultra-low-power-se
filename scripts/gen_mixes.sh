#!/usr/bin/env bash
set -euo pipefail

L="data/lists"
PY=python

# creează output dirs
mkdir -p data/prepared/train \
         data/prepared/dev \
         data/prepared/test_challenge/unseen_noises \
         data/prepared/test_challenge/reverb \
         data/prepared/test_challenge/codec_opus/16kbps \
         data/prepared/test_challenge/codec_opus/24kbps \
         data/prepared/test_challenge/clipping/hard \
         data/prepared/test_challenge/clipping/soft \
         data/prepared/test_challenge/streaming/manifests \
         data/prepared/test_standard/manifests

# --- TRAIN ---
if [ ! -s "$L/noise_train.txt" ]; then
  echo "[ERR] Lista de zgomote pentru TRAIN e goală. Populează datasets/DEMAND sau datasets/noises_custom și re-rulează scripts/build_lists.sh."
  exit 1
fi

$PY datasets/mixgen.py \
  --clean-list "$L/train_clean.txt" \
  --noise-list "$L/noise_train.txt" \
  --rir-list   "$L/rir_list.txt" \
  --out-dir    data/prepared/train \
  --snr 0 5 10 \
  --reverb-prob 0.3 \
  --n-mixes 50000 \
  --sr 16000 \
  --seed 1337

# --- VAL ---
$PY datasets/mixgen.py \
  --clean-list "$L/dev_clean.txt" \
  --noise-list "$L/noise_dev.txt" \
  --out-dir    data/prepared/dev \
  --snr 0 5 10 \
  --n-mixes 2000 \
  --sr 16000 \
  --seed 2025

# --- CHALLENGE: zgomote non-văzute ---
# folosim același clean ca la val; *necesită* un fișier separat cu zgomote „unseen”.
UNSEEN="data/lists/noise_unseen.txt"
if [ -s "$UNSEEN" ]; then
  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --out-dir    data/prepared/test_challenge/unseen_noises \
    --snr -5 0 5 10 \
    --n-mixes 4000 \
    --sr 16000 \
    --seed 404
else
  echo "[WARN] Nu există $UNSEEN — sar peste challenge/unseen_noises. Creează-l cu zgomote diferite de train/dev."
fi

# --- CHALLENGE: reverberație (folosește RIRs dacă există) ---
if [ -s "$L/rir_list.txt" ] && [ -s "$UNSEEN" ]; then
  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --rir-list   "$L/rir_list.txt" \
    --out-dir    data/prepared/test_challenge/reverb \
    --snr 0 5 10 \
    --reverb-prob 1.0 \
    --n-mixes 2000 \
    --sr 16000 \
    --seed 405
else
  echo "[WARN] Fără RIRs sau fără noise_unseen.txt – sar peste challenge/reverb."
fi

# --- CHALLENGE: codec OPUS ---
if [ -s "$UNSEEN" ]; then
  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --out-dir    data/prepared/test_challenge/codec_opus/16kbps \
    --snr 0 5 10 --codec opus_16 \
    --n-mixes 2000 --sr 16000 --seed 406

  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --out-dir    data/prepared/test_challenge/codec_opus/24kbps \
    --snr 0 5 10 --codec opus_24 \
    --n-mixes 2000 --sr 16000 --seed 407
fi

# --- CHALLENGE: clipping ---
if [ -s "$UNSEEN" ]; then
  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --out-dir    data/prepared/test_challenge/clipping/hard \
    --snr 0 5 10 --clipping hard \
    --n-mixes 1000 --sr 16000 --seed 408

  $PY datasets/mixgen.py \
    --clean-list "$L/dev_clean.txt" \
    --noise-list "$UNSEEN" \
    --out-dir    data/prepared/test_challenge/clipping/soft \
    --snr 0 5 10 --clipping soft \
    --n-mixes 1000 --sr 16000 --seed 409
fi

echo "[OK] Train/Dev și Challenge (acolo unde a fost posibil) au fost generate."
