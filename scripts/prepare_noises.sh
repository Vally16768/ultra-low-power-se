#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DEMAND_ROOT:=/data/DEMAND}"
NOISE_DIR="data/noise"
mkdir -p "$NOISE_DIR/demand" "$NOISE_DIR/synth"

# 1) DEMAND: doar listăm fișierele .wav existente
if [[ -d "$DEMAND_ROOT" ]]; then
  # Poți organiza train/dev/test după directoarele tale
  find "$DEMAND_ROOT" -type f -name "*.wav" | sort > data/lists/noise_all.txt
  # Heuristici simple pentru split (80/10/10) dacă nu ai subfoldere dedicate:
  total=$(wc -l < data/lists/noise_all.txt)
  train_n=$(( total*80/100 ))
  dev_n=$(( total*10/100 ))
  head -n $train_n data/lists/noise_all.txt > data/lists/noise_train.txt
  tail -n +$((train_n+1)) data/lists/noise_all.txt | head -n $dev_n > data/lists/noise_dev.txt
  tail -n +$((train_n+dev_n+1)) data/lists/noise_all.txt > data/lists/noise_test.txt
else
  echo "[prepare_noises] WARNING: DEMAND_ROOT not found ($DEMAND_ROOT). Using only synthetic noises."
  : > data/lists/noise_train.txt
  : > data/lists/noise_dev.txt
  : > data/lists/noise_test.txt
fi

# 2) Zgomote sintetice (white/pink/brown/babble)
python datasets/make_synth_noises.py
find data/noise/synth -type f -name "*.wav" | sort > data/lists/noise_synth.txt

# 3) noise_unseen: alege subset sintetice + (opțional) subset DEMAND nefolosit
# Simplu: folosim doar sintetice ca "unseen"
cp data/lists/noise_synth.txt data/lists/noise_unseen.txt

# 4) RIR list (opțional)
: "${RIRS_ROOT:=/data/RIRS_NOISES}"
if [[ -d "$RIRS_ROOT" ]]; then
  find "$RIRS_ROOT" -type f -name "*.wav" | sort > data/lists/rir_list.txt
else
  : > data/lists/rir_list.txt
fi

echo "[prepare_noises] lists:"
for f in noise_train.txt noise_dev.txt noise_test.txt noise_unseen.txt rir_list.txt; do
  echo -n "  $f: "; wc -l "data/lists/$f"
done
