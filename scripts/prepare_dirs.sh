# scripts/prepare_dirs.sh
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/lists \
         data/clean_wav/libri \
         data/prepared/train \
         data/prepared/dev \
         data/prepared/test_standard/manifests \
         data/prepared/test_challenge/unseen_noises \
         data/prepared/test_challenge/reverb \
         data/prepared/test_challenge/codec_opus/16kbps \
         data/prepared/test_challenge/codec_opus/24kbps \
         data/prepared/test_challenge/clipping/hard \
         data/prepared/test_challenge/clipping/soft \
         data/prepared/test_challenge/streaming/manifests
echo "[OK] Structura creată."
