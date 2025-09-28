# Ultra-Low-Power Speech Enhancement

This repository contains a **modular, end-to-end pipeline** for training, exporting, and benchmarking ultra‑low‑power speech enhancement models (Tiny Mamba‑U‑Net stub, DCCRN, LSTM baseline).

## Architecture at a Glance

```
ultra-low-power-se/
├─ configs/                 # YAML configs (data, model, train, export)
├─ datasets/                # Data loaders & mix generation (LibriSpeech, VBD, VoxCeleb)
├─ deploy/                  # Export to ONNX + sanity checks; edge runners (GAP9/STM32)
├─ runners/                 # CLI “runnables”: train / infer / export / evaluate / score
├─ scripts/                 # Shell helpers: prepare data, build lists, gen mixes
├─ se_cli/                  # CLI entrypoint + config merge
├─ se_models/               # Models: mamba_unet (stub), dccrn, lstm_baseline
├─ tests/                   # Smoke tests for config/metrics/models
├─ metrics/, eval/, reports/ (optional usage)
└─ artifacts/               # Outputs (checkpoints, ONNX, logs)
```

## Quickstart

```bash
make setup           # create .venv and install deps
make train           # train stub model on dummy data (smoke test)
make export          # export ONNX to artifacts/export/
make onnx-sanity     # run an ONNX forward pass
```

> **Note**: Install PyTorch that matches your system separately if needed.

## CLI

The CLI is a thin wrapper over `runners/*`:

```bash
python -m se_cli.cli train   --config configs/exp_mamba_unet.yaml
python -m se_cli.cli export  --config configs/exp_mamba_unet.yaml -o export.name=mamba_unet_v0
python -m se_cli.cli enhance --config configs/exp_mamba_unet.yaml -o inference.in_wav=path.wav
```

See per-module READMEs in each folder for details.