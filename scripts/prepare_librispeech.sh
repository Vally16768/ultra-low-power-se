#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${LIBRISPEECH_ROOT:=/data/LibriSpeech}"

OUT_ROOT="data/clean_wav/libri"
mkdir -p "$OUT_ROOT"

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
  if [[ -s "$dst" ]]; then continue; fi
  ffmpeg -nostdin -y -loglevel error -i "$src" -ac 1 -ar 16000 "$dst"
done

# Liste train/dev (din structura standard LibriSpeech)
find "$OUT_ROOT/train-clean-360" -type f -name "*.wav" | sort > data/lists/train_clean.txt || true
if [[ ! -s data/lists/train_clean.txt ]]; then
  # fallback: orice 'train-clean-*'
  find "$OUT_ROOT" -type f -path "*/train-clean-*/*.wav" | sort > data/lists/train_clean.txt || true
fi

find "$OUT_ROOT/dev-clean" -type f -name "*.wav" | sort > data/lists/dev_clean.txt || true
if [[ ! -s data/lists/dev_clean.txt ]]; then
  # fallback: orice 'dev-*'
  find "$OUT_ROOT" -type f -path "*/dev-*/*.wav" | sort > data/lists/dev_clean.txt || true
fi

echo "[prepare_librispeech] lists:"
wc -l data/lists/train_clean.txt || true
wc -l data/lists/dev_clean.txt   || true
