#!/usr/bin/env bash
set -euo pipefail

DEMAND_DIR="datasets/DEMAND"
CUSTOM_DIR="datasets/noises_custom"
RIRS_DIR="datasets/RIRs"

resample_tree () {
  local ROOT="$1"
  [ -d "$ROOT" ] || { echo "[WARN] Lipsește $ROOT – sar."; return 0; }
  echo "[INFO] Resampling la 16k/mono în $ROOT (înlocuiește extensia cu .wav dacă e diferită)..."
  # convertim TOATE fișierele (wav/flac/mp3 etc.) în wav 16k mono *în același loc*
  find "$ROOT" -type f ! -name ".*" -print0 | while IFS= read -r -d '' f; do
    ext="${f##*.}"
    out="${f%.*}.wav"
    if [[ "$ext" != "wav" ]]; then
      ffmpeg -nostdin -loglevel error -y -i "$f" -ar 16000 -ac 1 "$out"
    else
      # dacă e deja wav, forțăm ar/mono în loc (creează fișier temporar)
      tmp="${f}.tmp.wav"
      ffmpeg -nostdin -loglevel error -y -i "$f" -ar 16000 -ac 1 "$tmp"
      mv -f "$tmp" "$f"
    fi
  done
  echo "[OK] $ROOT"
}

resample_tree "$DEMAND_DIR"
resample_tree "$CUSTOM_DIR"

if [ -d "$RIRS_DIR" ]; then
  echo "[INFO] RIRs în $RIRS_DIR (nu schimb nimic)."
else
  echo "[WARN] Lipsește $RIRS_DIR – reverberația va fi dezactivată."
fi
