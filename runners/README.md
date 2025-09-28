# runners – Entry points

- `train.py`: minimal train loop using a dummy dataloader (for smoke test).
- `infer.py`: runs a forward pass on a WAV (or random noise) and writes `out_wav`.
- `export.py`: calls `deploy/export_onnx.py` with config.
- `evaluate.py`, `score.py`: placeholders (wire up your datasets/metrics).

### Commands
```bash
python -m se_cli.cli train --config configs/exp_mamba_unet.yaml
python -m se_cli.cli enhance --config configs/exp_mamba_unet.yaml -o inference.seconds=2
python -m se_cli.cli export --config configs/exp_mamba_unet.yaml
```
