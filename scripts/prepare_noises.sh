#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROOT="data"
DEMAND="$ROOT/noise/demand"
CUSTOM="$ROOT/noise/custom"
SYNTH="$ROOT/noise/synth"
DEV_DIR="$ROOT/noise/dev"
UNSEEN_DIR="$ROOT/noise/unseen"
LISTS="$ROOT/lists"
mkdir -p "$LISTS"

has_wav() { find "$1" -type f -iname "*.wav" 2>/dev/null | head -n1 >/dev/null; }

# --- (opțional) generează synth dacă nu ai NICIUN zgomot ---
SYNTH_CANDIDATES=(
  "make_synth_noises.py"
  "scripts/make_synth_noises.py"
  "datasets/make_synth_noises.py"
  "data/datasets/make_synth_noises.py"
)
if ! has_wav "$DEMAND" && ! has_wav "$CUSTOM" && ! has_wav "$SYNTH"; then
  for c in "${SYNTH_CANDIDATES[@]}"; do
    if [[ -f "$c" ]]; then
      echo "[prepare_noises] generating synth noises via $c"
      python "$c"
      break
    fi
  done
fi

# ---------- NOISE TRAIN ----------
: > "$LISTS/noise_train.txt"
for D in "$DEMAND" "$CUSTOM" "$SYNTH"; do
  if [[ -d "$D" ]]; then
    find "$D" -type f -iname "*.wav" | sort >> "$LISTS/noise_train.txt"
  fi
done

# ---------- NOISE DEV ----------
if [[ -d "$DEV_DIR" ]]; then
  find "$DEV_DIR" -type f -iname "*.wav" | sort > "$LISTS/noise_dev.txt"
else
  # fallback: primele 200 din train
  if [[ -s "$LISTS/noise_train.txt" ]]; then
    head -n 200 "$LISTS/noise_train.txt" > "$LISTS/noise_dev.txt"
  else
    : > "$LISTS/noise_dev.txt"
  fi
fi

# ---------- NOISE UNSEEN ----------
if [[ -d "$UNSEEN_DIR" ]]; then
  find "$UNSEEN_DIR" -type f -iname "*.wav" | sort > "$LISTS/noise_unseen.txt"
else
  if [[ -s "$LISTS/noise_train.txt" ]]; then
    comm -23 <(sort "$LISTS/noise_train.txt") <(sort "$LISTS/noise_dev.txt") | shuf -n 50 > "$LISTS/noise_unseen.txt" || true
  else
    : > "$LISTS/noise_unseen.txt"
  fi
fi

# ---------- NOISE TEST (alias dev, pentru compat) ----------
cp -f "$LISTS/noise_dev.txt" "$LISTS/noise_test.txt"

# ---------- RIRs ----------
RIRS="${RIRS_ROOT:-$ROOT/rirs}"
if [[ -d "$RIRS" ]]; then
  find "$RIRS" -type f -iname "*.wav" | sort > "$LISTS/rir_list.txt" || true
else
  : > "$LISTS/rir_list.txt" || true
fi

# ---------- Rezumat ----------
for f in noise_train.txt noise_dev.txt noise_test.txt noise_unseen.txt rir_list.txt; do
  n=0; [[ -f "$LISTS/$f" ]] && n=$(wc -l < "$LISTS/$f" || echo 0)
  printf "  %-16s: %s %s/%s\n" "$f" "$n" "$LISTS" "$f"
done

# ---------- Sanity ----------
if [[ ! -s "$LISTS/noise_train.txt" ]]; then
  echo "[prepare_noises] ERROR: nu am găsit zgomote pentru train în $DEMAND / $CUSTOM / $SYNTH" >&2
  exit 4
fi

echo "[prepare_noises] OK"
