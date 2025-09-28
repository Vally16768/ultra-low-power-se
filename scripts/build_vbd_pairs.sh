#!/usr/bin/env bash
set -euo pipefail
# Construieste perechi pentru VoiceBank-DEMAND (test standard)
# Folosește layout-ul clasic VBD:
#   <VBD_ROOT>/clean_testset_wav/*.wav
#   <VBD_ROOT>/noisy_testset_wav/*.wav (sau DEMAND/testset_wav)
# Setează VBD_ROOT în .env.local dacă diferă.

: "${VBD_ROOT:=data/VoiceBank}"

OUT="data/lists/vbd_pairs_test.csv"
mkdir -p "$(dirname "$OUT")"

CLEAN_DIR="${VBD_ROOT}/clean_testset_wav"
NOISY_DIR="${VBD_ROOT}/noisy_testset_wav"
ALT_NOISY_DEMAND="data/DEMAND/testset_wav"  # fallback uzual în unele repo-uri

if [[ ! -d "$CLEAN_DIR" ]]; then
  echo "[build_vbd_pairs] WARNING: $CLEAN_DIR not found. Skipping pairs."
  : > "$OUT"
  exit 0
fi

# Manifest (noisy,clean,meta) — meta gol pentru VBD
echo "noisy,clean,meta" > "$OUT"

shopt -s nullglob
for c in "$CLEAN_DIR"/*.wav; do
  base="$(basename "$c")"
  n1="$NOISY_DIR/$base"
  n2="$ALT_NOISY_DEMAND/$base"
  if [[ -f "$n1" ]]; then
    echo "$n1,$c," >> "$OUT"
  elif [[ -f "$n2" ]]; then
    echo "$n2,$c," >> "$OUT"
  else
    echo "[build_vbd_pairs] WARN: No noisy match for $base" >&2
  fi
done

echo "[build_vbd_pairs] wrote: $OUT"
