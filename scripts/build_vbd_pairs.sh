# scripts/build_vbd_pairs.sh
#!/usr/bin/env bash
set -euo pipefail

VBD="datasets/VoiceBank_DEMAND"
OUT_DIR="data/prepared/test_standard/manifests"
mkdir -p "$OUT_DIR"
PAIRS="$OUT_DIR/pairs.csv"

NOISY_TEST="$VBD/noisy_testset_wav"
CLEAN_TEST="$VBD/clean_testset_wav"

if [ ! -d "$NOISY_TEST" ] || [ ! -d "$CLEAN_TEST" ]; then
  echo "[ERR] VoiceBank_DEMAND nu e în structura așteptată: $NOISY_TEST / $CLEAN_TEST"
  exit 1
fi

echo "noisy,clean" > "$PAIRS"

# mapăm fiecare fișier noisy la clean cu același nume
while IFS= read -r -d '' n; do
  base="$(basename "$n")"
  c="$CLEAN_TEST/$base"
  if [ -f "$c" ]; then
    echo "$n,$c" >> "$PAIRS"
  else
    echo "[WARN] Lipsește clean pentru $base"
  fi
done < <(find "$NOISY_TEST" -type f -name "*.wav" -print0 | sort -z)

echo "[OK] Scris $PAIRS ($(($(wc -l < "$PAIRS")-1)) perechi)."
