#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DEMAND_ROOT:=/data/DEMAND}"
NOISE_DIR="data/noise"
mkdir -p "$NOISE_DIR/demand" "$NOISE_DIR/synth" data/lists

# --- 1) DEMAND: facem lista completă (wav/flac), dacă există ---
if [[ -d "$DEMAND_ROOT" ]]; then
  # listă stabilă, sortată
  find "$DEMAND_ROOT" -type f \( -iname "*.wav" -o -iname "*.flac" \) \
    | sort > data/lists/noise_all.txt || true
else
  : > data/lists/noise_all.txt
fi

total=$( (wc -l < data/lists/noise_all.txt) 2>/dev/null || echo 0 )
train_n=$(( total * 80 / 100 ))
dev_n=$(( total * 10 / 100 ))
# test_n = restul
test_n=$(( total - train_n - dev_n ))

# Evităm SIGPIPE și folosim awk (nu `tail|head`)
if (( total > 0 )); then
  awk -v n="$train_n"          'NR<=n'  data/lists/noise_all.txt > data/lists/noise_train.txt
  awk -v o="$train_n" -v n="$dev_n"     'NR>o && NR<=o+n' data/lists/noise_all.txt > data/lists/noise_dev.txt
  awk -v o="$((train_n+dev_n))"         'NR>o' data/lists/noise_all.txt > data/lists/noise_test.txt
else
  : > data/lists/noise_train.txt
  : > data/lists/noise_dev.txt
  : > data/lists/noise_test.txt
fi

# --- 2) Zgomote sintetice (white/pink/brown/babble) ---
# Detectăm corect calea scriptului (poate fi în repo la rădăcină sau sub data/)
SYNTH_SCRIPT="datasets/make_synth_noises.py"
if [[ ! -f "$SYNTH_SCRIPT" ]]; then
  if [[ -f "data/datasets/make_synth_noises.py" ]]; then
    SYNTH_SCRIPT="data/datasets/make_synth_noises.py"
  else
    echo "[prepare_noises] ERROR: nu găsesc datasets/make_synth_noises.py" >&2
    exit 5
  fi
fi

python "$SYNTH_SCRIPT"

# listă sintetice
find data/noise/synth -type f -name "*.wav" | sort > data/lists/noise_synth.txt || : 

# --- 3) Unseen noises: dacă există sintetice, folosește-le ca unseen ---
if [[ -s data/lists/noise_synth.txt ]]; then
  cp -f data/lists/noise_synth.txt data/lists/noise_unseen.txt
else
  # fallback: unseen = test din DEMAND
  cp -f data/lists/noise_test.txt data/lists/noise_unseen.txt || : 
fi

# --- 4) RIR list (dacă există) ---
: "${RIRS_ROOT:=/data/RIRS_NOISES}"
if [[ -d "$RIRS_ROOT" ]]; then
  find "$RIRS_ROOT" -type f -name "*.wav" | sort > data/lists/rir_list.txt
else
  : > data/lists/rir_list.txt
fi

echo "[prepare_noises] lists:"
for f in noise_train.txt noise_dev.txt noise_test.txt noise_unseen.txt rir_list.txt; do
  printf "  %s: " "$f"
  (wc -l "data/lists/$f" 2>/dev/null) || echo "0 data/lists/$f"
done
