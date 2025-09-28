# se_cli – Command-Line Interface

`se_cli` provides two things:
- `config.py`: YAML loader + hierarchical `-o key.path=value` overrides.
- `cli.py`: dispatch to `runners/*`.

### Run
```bash
python -m se_cli.cli train --config configs/exp_mamba_unet.yaml -o train.epochs=3
```
Expected: a run folder under `artifacts/exp/<name>/YYYYmmdd-HHMMSS` with `ckpt/model.ckpt` and `logs/`.
