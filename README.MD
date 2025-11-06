# A) Directly specify the exact CSV (most explicit)
export ULPSE_TRAIN_CSV="/home/vpopescu/projects/ultra-low-power-se/datasets/datasets/voicebank-demand/16k/train.csv"

# B) Or just the folder that contains train.csv
export ULPSE_DATA="/home/vpopescu/projects/ultra-low-power-se/datasets/datasets/voicebank-demand/16k"

# C) Or fix repo root if discovery picks the wrong one
export ULPSE_ROOT="/home/vpopescu/projects/ultra-low-power-se"

python core/features/create_training_data.py \
  --csv datasets/datasets/voicebank-demand/16k/train.csv \
  --outdir runs/feat_cache \
  --subset-name train \
  --jobs 25

python core/features/create_training_data.py \
  --csv datasets/datasets/voicebank-demand/16k/test.csv \
  --outdir runs/feat_cache \
  --subset-name test \
  --jobs 25

nohup python core/train.py \
  --train-manifest runs/feat_cache/train_manifest.csv \
  --test-manifest  runs/feat_cache/test_manifest.csv \
  --train-stats    runs/feat_cache/train_feature_stats.npz \
  --outdir         runs/exp_mask_head_v2 \
  --batch-size 32 --epochs 1000 --lr 1e-3 \
  > runs/exp_mask_head_v2/train.log 2>&1 &
echo $! > runs/exp_mask_head_v2/train.pid


nohup bash download-dns-challenge-2.sh > download.log 2>&1 &