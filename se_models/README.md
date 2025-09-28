# se_models – Models

- `mamba_unet/`: Tiny Mamba‑U‑Net **stub** (Conv1d U-Net) for smoke tests.
- `dccrn/`, `lstm_baseline/`: placeholders (fill with real implementations).

### Config hook
The runner imports `se_models.<name>.model:build_model(cfg)` and expects a `torch.nn.Module`.

### Export
```bash
python -m se_cli.cli export --config configs/exp_mamba_unet.yaml
```
