# A) Directly specify the exact CSV (most explicit)
export ULPSE_TRAIN_CSV="/home/vpopescu/projects/ultra-low-power-se/datasets/datasets/voicebank-demand/16k/train.csv"

# B) Or just the folder that contains train.csv
export ULPSE_DATA="/home/vpopescu/projects/ultra-low-power-se/datasets/datasets/voicebank-demand/16k"

# C) Or fix repo root if discovery picks the wrong one
export ULPSE_ROOT="/home/vpopescu/projects/ultra-low-power-se"

nohup python -m core.train > train.log 2>&1 &


  python core/features/create_training_data.py \
    --csv datasets/datasets/voicebank-demand/16k/train.csv \
    --outdir runs/feat_cache \
    --subset-name train \
    --mel-ceps 0 \
    --jobs 8


python core/train.py \
  --train-manifest runs/feat_cache/train_manifest.csv \
  --test-manifest  runs/feat_cache/test_manifest.csv \
  --outdir runs/exp_mask_head_v1 \
  --batch-size 16 --epochs 100 --lr 1e-3 --mel-ceps 0