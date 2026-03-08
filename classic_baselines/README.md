# Classic Baselines

This subproject runs traditional speech-enhancement baselines on the same
VoiceBank+DEMAND split used by the neural baseline.

## Methods

- `noisy`
- `wiener`
- `spectral_subtraction`
- `spectral_gating`
- `mmse_stsa`
- `logmmse`
- `pca_subspace`
- `pca_wiener`

## Typical usage

```bash
python -m classic_baselines.pca \
  --train_csv datasets/datasets/voicebank-demand/16k/train.csv \
  --out classic_baselines/artifacts/voicebank_demand/pca_model.npz

python -m classic_baselines.run_suite \
  --train_csv datasets/datasets/voicebank-demand/16k/train.csv \
  --test_csv datasets/datasets/voicebank-demand/16k/test.csv \
  --out_root classic_baselines/runs/voicebank_demand
```
