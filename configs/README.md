# configs – YAML Examples

- `exp_mamba_unet.yaml`: tiny training/export config for the stub model.
- `default.yaml`, `local.dev.yaml`: inherit/override patterns.
- `export_template.yaml`: minimal ONNX export config.

### Edit on the fly
```bash
python -m se_cli.cli train --config configs/exp_mamba_unet.yaml -o train.lr=3e-4 train.epochs=2
```
