# deploy – Export & Edge Runners

- `export_onnx.py`: builds model from `cfg` and exports dynamic‑T ONNX.
- `sanity_onnx.py`: tests ONNX forward with onnxruntime.
- `edge/`: runners for GAP9 / STM32 (skeletons).

### Export
```bash
python -m se_cli.cli export --config configs/exp_mamba_unet.yaml -o export.name=mamba_unet_v0
python deploy/sanity_onnx.py artifacts/export/mamba_unet_v0.onnx
```
Expected: `[export] ONNX saved to: artifacts/export/mamba_unet_v0.onnx` and sanity message.
