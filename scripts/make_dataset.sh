#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIBRISPEECH_ROOT="${1:-}"
DEMAND_ROOT="${2:-}"
RIRS_ROOT="${3:-}"

# Parse args în format --key value
LIBRISPEECH_ROOT=""
DEMAND_ROOT=""
RIRS_ROOT=""
VERIFY_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --librispeech) LIBRISPEECH_ROOT="$2"; shift 2;;
    --demand)      DEMAND_ROOT="$2"; shift 2;;
    --rirs)        RIRS_ROOT="$2"; shift 2;;
    --verify-only) VERIFY_ONLY="1"; shift 1;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

export LIBRISPEECH_ROOT DEMAND_ROOT RIRS_ROOT

# 0) Dependențe minime – fail fast dacă lipsesc
command -v ffmpeg >/dev/null || { echo "Missing ffmpeg"; exit 3; }
python -c "import numpy,soundfile,resampy" 2>/dev/null || {
  echo "Missing Python deps: pip install numpy soundfile resampy"; exit 3; }

# 1) Pregătește arborescența
bash scripts/prepare_dirs.sh

if [[ "$VERIFY_ONLY" == "1" ]]; then
  # verifică train/dev/challenge (dacă există)
  ok=0
  fail=0
  check_dir() {
    local d="$1"
    if [[ -d "$d" ]]; then
      python datasets/mixgen.py \
        --clean-list data/lists/train_clean.txt \
        --noise-list data/lists/noise_train.txt \
        --out-dir "$d" \
        --snr -5 0 5 10 15 \
        --verify-only && ok=$((ok+1)) || fail=$((fail+1))
    fi
  }
  check_dir "data/prepared/train_mixes"
  check_dir "data/prepared/dev_mixes"
  for d in data/prepared/test_challenge/*; do
    [[ -d "$d" ]] && python datasets/mixgen.py \
      --clean-list data/lists/dev_clean.txt \
      --noise-list data/lists/noise_unseen.txt \
      --out-dir "$d" \
      --snr -5 0 5 10 \
      --verify-only && ok=$((ok+1)) || fail=$((fail+1))
  done
  echo "[verify] OK=$ok FAIL=$fail"
  exit $([[ "$fail" == "0" ]] && echo 0 || echo 1)
fi

# 2) Convertește LibriSpeech în mono 16k + face liste (idempotent)
bash scripts/prepare_librispeech.sh

# 3) Pregătește zgomote (DEMAND + sintetice) + liste (idempotent)
bash scripts/prepare_noises.sh

# 4) VoiceBank-DEMAND pairs (idempotent)
bash scripts/build_vbd_pairs.sh

# 5) Generează mixuri (idempotent, reia doar lipsurile)
bash scripts/gen_mixes.sh

echo "[datasets] DONE ✅"
