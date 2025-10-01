#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/{clean_wav/libri,noise/{demand,synth},lists,prepared}
mkdir -p data/prepared/{train_mixes,dev_mixes,test_challenge}
echo "[prepare_dirs] OK"
