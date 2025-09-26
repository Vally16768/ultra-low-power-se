#!/usr/bin/env bash
set -euo pipefail

LIBRI_WAV_ROOT="data/clean_wav/libri"
DEMAND_DIR="datasets/DEMAND"
CUSTOM_DIR="datasets/noises_custom"

LISTS_DIR="data/lists"
mkdir -p "$LISTS_DIR"

# --- clean speech ---
# Train: preferăm test-other + test-clean (ca fallback), pentru că nu ai train-*; dev-* le folosim pentru val.
> "$LISTS_DIR/train_clean.txt"
for SPLIT in test-other test-clean; do
  if [ -d "$LIBRI_WAV_ROOT/$SPLIT" ]; then
    find "$LIBRI_WAV_ROOT/$SPLIT" -type f -name "*.wav" | sort >> "$LISTS_DIR/train_clean.txt"
  fi
done
if [ ! -s "$LISTS_DIR/train_clean.txt" ]; then
  echo "[ERR] Nu am găsit clean WAV pentru TRAIN (test-clean/other convertite). Rulează scripts/prepare_librispeech.sh."
  exit 1
fi

# Val: dev-clean (+ opțional dev-other pentru val extins)
if [ -d "$LIBRI_WAV_ROOT/dev-clean" ]; then
  find "$LIBRI_WAV_ROOT/dev-clean" -type f -name "*.wav" | sort > "$LISTS_DIR/dev_clean.txt"
else
  echo "[ERR] Nu găsesc $LIBRI_WAV_ROOT/dev-clean"; exit 1
fi

# --- noises split (train/dev) ---
TMP_NOISE_LIST="$(mktemp)"
{ [ -d "$DEMAND_DIR" ] && find "$DEMAND_DIR" -type f -name "*.wav"; \
  [ -d "$CUSTOM_DIR" ] && find "$CUSTOM_DIR" -type f -name "*.wav"; } \
  | sort > "$TMP_NOISE_LIST" || true

if [ ! -s "$TMP_NOISE_LIST" ]; then
  echo "[WARN] Nu există zgomote (DEMAND/noises_custom). Continuăm, dar mixgen va eșua fără ele."
  : > "$LISTS_DIR/noise_train.txt"
  : > "$LISTS_DIR/noise_dev.txt"
else
  # împărțim determinist 80/20
  SEED=1337
  mapfile -t ALL_NOISES < <(shuf --random-source=<(yes $SEED) "$TMP_NOISE_LIST")
  N=${#ALL_NOISES[@]}
  CUT=$(( (N*80 + 99)/100 ))
  printf "%s\n" "${ALL_NOISES[@]:0:CUT}"   > "$LISTS_DIR/noise_train.txt"
  printf "%s\n" "${ALL_NOISES[@]:CUT}"     > "$LISTS_DIR/noise_dev.txt"
  echo "[OK] zgomote: train=$(wc -l < "$LISTS_DIR/noise_train.txt")  dev=$(wc -l < "$LISTS_DIR/noise_dev.txt")"
fi

rm -f "$TMP_NOISE_LIST"

# --- RIRs (opțional) ---
RIRS_DIR="datasets/RIRs"
if [ -d "$RIRS_DIR" ]; then
  find "$RIRS_DIR" -type f -name "*.wav" | sort > "$LISTS_DIR/rir_list.txt"
  echo "[OK] RIR list construit."
else
  : > "$LISTS_DIR/rir_list.txt"
  echo "[WARN] Fără RIRs – fișier gol creat."
fi

# Rezumat
echo "==== LISTE ===="
for f in train_clean.txt dev_clean.txt noise_train.txt noise_dev.txt rir_list.txt; do
  printf "%-18s %7d\n" "$f" "$(wc -l < "$LISTS_DIR/$f" 2>/dev/null || echo 0)"
done
