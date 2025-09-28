#!/usr/bin/env bash
set -euo pipefail

# TRAIN
python datasets/mixgen.py \
  --clean-list data/lists/train_clean.txt \
  --noise-list data/lists/noise_train.txt \
  --rir-list   data/lists/rir_list.txt \
  --out-dir    data/prepared/train_mixes \
  --snr -5 0 5 10 15 \
  --sr 16000 \
  --n-mixes 50000 \
  --reverb-prob 0.35 \
  --codec none \
  --clipping none \
  --segment-min 2.0 --segment-max 8.0 \
  --seed 1337

# DEV
python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_dev.txt \
  --rir-list   data/lists/rir_list.txt \
  --out-dir    data/prepared/dev_mixes \
  --snr -5 0 5 10 \
  --sr 16000 \
  --n-mixes 4000 \
  --reverb-prob 0.3 \
  --codec none \
  --clipping none \
  --segment-min 2.0 --segment-max 8.0 \
  --seed 2025

# CHALLENGE – unseen
python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_unseen.txt \
  --out-dir    data/prepared/test_challenge/unseen_noises \
  --snr -5 0 5 10 \
  --sr 16000 \
  --n-mixes 4000 \
  --segment-min 2.0 --segment-max 6.0 \
  --seed 404

# CHALLENGE – OPUS
python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_unseen.txt \
  --out-dir    data/prepared/test_challenge/opus16 \
  --snr 0 5 10 \
  --sr 16000 \
  --n-mixes 2000 \
  --codec opus_16 \
  --seed 405

python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_unseen.txt \
  --out-dir    data/prepared/test_challenge/opus24 \
  --snr 0 5 10 \
  --sr 16000 \
  --n-mixes 2000 \
  --codec opus_24 \
  --seed 406

# CHALLENGE – clipping
python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_unseen.txt \
  --out-dir    data/prepared/test_challenge/clipping_hard \
  --snr -5 0 5 \
  --sr 16000 \
  --n-mixes 1500 \
  --clipping hard \
  --seed 407

python datasets/mixgen.py \
  --clean-list data/lists/dev_clean.txt \
  --noise-list data/lists/noise_unseen.txt \
  --out-dir    data/prepared/test_challenge/clipping_soft \
  --snr -5 0 5 \
  --sr 16000 \
  --n-mixes 1500 \
  --clipping soft \
  --seed 408
