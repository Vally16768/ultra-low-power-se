#!/usr/bin/env bash
set -euo pipefail

LSB_ROOT="datasets/LibriSpeech"
OUT_ROOT="data/clean_wav/libri"

command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "[ERR] ffmpeg nu este instalat."; exit 1; }

if [ ! -d "$LSB_ROOT" ]; then
  echo "[ERR] Nu găsesc $LSB_ROOT"; exit 1
fi

echo "[INFO] Convert FLAC -> WAV 16k mono din $LSB_ROOT în $OUT_ROOT ..."
mapfile -d '' FLACS < <(find "$LSB_ROOT" -type f -name "*.flac" -print0)
if [ ${#FLACS[@]} -eq 0 ]; then
  echo "[WARN] Nu există fișiere .flac în $LSB_ROOT"; exit 0
fi

for f in "${FLACS[@]}"; do
  rel="${f#${LSB_ROOT}/}"
  out="$OUT_ROOT/${rel%.flac}.wav"
  mkdir -p "$(dirname "$out")"
  ffmpeg -nostdin -loglevel error -y -i "$f" -ar 16000 -ac 1 "$out"
done

echo "[OK] Conversia s-a terminat."
