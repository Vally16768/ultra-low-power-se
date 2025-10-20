# Ultra SE (Slim)

A slim, modular refactor of your on-the-fly IRM training script with **fail-fast checks**
and **progress bars everywhere** (manifests, preloading, training, quick eval).

## Layout

```text
ultra_se_slim/
  __init__.py
  config.py        # central defaults
  features.py      # STFT helpers + masks
  data.py          # TrainSequence & ValSequence (with preload progress)
  model.py         # tiny MLP-IRM model builder
  eval_utils.py    # quick SNR improvement eval (with progress)
  train.py         # CLI entry-point
```

## Install

```bash
pip install numpy scipy soundfile tensorflow scikeras tqdm
```

Your existing `augment` package is still used. We import `augment.base.apply_chain` at batch time.

## Run

```bash
python -m ultra_se_slim.train       --train manifests/train.csv       --val manifests/val_voicebank.csv       --epochs 10 --steps_per_epoch 200 --val_steps 20
```

* Keras prints epoch progress.
* Preloading and evaluation show `tqdm` progress bars.
* Any unexpected condition raises immediately (sample rate mismatch, empty manifests, missing files).
* Best model checkpoint: `best_mask_mlp.keras` + `training_log.csv`.
```
