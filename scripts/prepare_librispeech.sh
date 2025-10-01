#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${LIBRISPEECH_ROOT:=/data/LibriSpeech}"

OUT_ROOT="data/clean_wav/libri"
mkdir -p "$OUT_ROOT" data/lists

if [[ ! -d "$LIBRISPEECH_ROOT" ]]; then
  echo "[prepare_librispeech] WARNING: LIBRISPEECH_ROOT not found ($LIBRISPEECH_ROOT). Skipping."
  exit 0
fi

# Convertim toate .flac în mono 16k (idempotent, skip dacă există)
find "$LIBRISPEECH_ROOT" -type f -name "*.flac" | while read -r src; do
  rel="${src#$LIBRISPEECH_ROOT/}"
  dst_dir="$OUT_ROOT/$(dirname "$rel")"
  dst="${dst_dir}/$(basename "${src%.flac}").wav"
  mkdir -p "$dst_dir"
  [[ -s "$dst" ]] && continue
  ffmpeg -nostdin -y -loglevel error -i "$src" -ac 1 -ar 16000 "$dst"
done

# Liste train/dev
# 1) train: căutăm train-clean-360 / 100; dacă lipsesc, folosim dev ca fallback
if find "$OUT_ROOT/train-clean-"* -maxdepth 1 -type d >/dev/null 2>&1; then
  find "$OUT_ROOT" -type f -path "*/train-clean-*/*.wav" | sort > data/lists/train_clean.txt || true
else
  # fallback: dacă nu există train, copiem dev pe post de train (numai pt. a nu pica pipeline-ul)
  : > data/lists/train_clean.txt
fi

# 2) dev
if [[ -d "$OUT_ROOT/dev-clean" ]]; then
  find "$OUT_ROOT/dev-clean" -type f -name "*.wav" | sort > data/lists/dev_clean.txt || true
else
  # fallback generic: orice dev-*
  find "$OUT_ROOT" -type f -path "*/dev-*/*.wav" | sort > data/lists/dev_clean.txt || true
fi

# dacă train e gol dar dev există, folosim dev și ca train (fallback controlat)
if [[ ! -s data/lists/train_clean.txt && -s data/lists/dev_clean.txt ]]; then
  cp -f data/lists/dev_clean.txt data/lists/train_clean.txt
  echo "[prepare_librispeech] NOTICE: train_clean.txt absent → using dev_clean.txt as train (fallback)."
fi

echo "[prepare_librispeech] lists:"
(wc -l data/lists/train_clean.txt 2>/dev/null) || echo "0 data/lists/train_clean.txt"
(wc -l data/lists/dev_clean.txt   2>/dev/null) || echo "0 data/lists/dev_clean.txt"
