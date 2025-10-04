# Ultra-Low-Power Speech Enhancement

A modular, end-to-end pipeline for training, exporting, and benchmarking ultra-low-power speech enhancement models (Tiny Mamba-U-Net stub, DCCRN, LSTM baseline).

✨ **Highlights**
- Config-first (YAML) for data/model/training/export
- Unified CLI: `train` / `eval` / `enhance` / `export`
- ONNX export + sanity & PT↔ONNX parity checks
- Quantization (dynamic INT8) for edge deployment
- Makefile with quick targets + pre-commit (ruff/mypy etc.)

---

## 📁 Project Structure

```
ultra-low-power-se/
├─ .github/workflows/        # CI
├─ ab/                       # experiments/ABX (optional)
├─ baselines/                # baseline models
├─ ci/                       # auxiliary CI configs
├─ configs/                  # YAML (data, model, train, export)
├─ datasets/                 # data loaders / mix generators
├─ deploy/                   # ONNX export, quantization, edge runners
├─ eval/, metrics/, reports/ # evaluation/metrics/reports (optional)
├─ lists/                    # list/manifest utilities
├─ runners/                  # train / infer / eval / score
├─ scripts/                  # helper shell scripts
├─ se_cli/                   # CLI entrypoint + config merge
├─ se_models/                # mamba_unet, dccrn, lstm_baseline
├─ tests/                    # smoke/unit tests
├─ tools/                    # parity checks, utilities
└─ artifacts/                # checkpoints, ONNX, logs
```

> Note: This structure reflects the branch `feature/ULP-3-stabilize-the-code` on GitHub.

---

## 🧰 Requirements
- Python 3.10 / 3.11
- PyTorch (install the version compatible with your system)
- `onnx` / `onnxruntime` (installed automatically via `make dev-setup`)
- (optional) CUDA for accelerated training

---

## 🚀 Installation & Setup

```bash
make setup        # creates .venv, install -e .
make dev-setup    # installs ruff, mypy, pytest, onnx, onnxruntime, pre-commit
```

Optional quality hooks:

```bash
make pre-commit-install   # installs pre-commit & commit-msg hooks
make pre-commit           # runs all hooks on the repo
```

If your Python was built without `sqlite3`, install `pysqlite3-binary` and inject it (see “sqlite fix” in issues/notes).

---

## 🎛️ Configuration

Main config example: `configs/exp_mamba_unet.yaml` — defines the experiment, model module, data (manifests), training, evaluation, and inference hyperparameters.
Manifests are CSV files with columns: `noisy`, `clean` (absolute/relative paths).

---

## 🏋️ Training

CLI (wrapper around `runners/train.py` via `se_cli`):

```bash
python -m se_cli.cli train --config configs/exp_mamba_unet.yaml
# or override specific options:
python -m se_cli.cli train --config configs/exp_mamba_unet.yaml   -o train.epochs=1 -o train.batch_size=2
```

or simply:

```bash
make train
```

---

## 🎧 Inference (offline)

```bash
python -m se_cli.cli enhance --config configs/exp_mamba_unet.yaml   -o inference.in_wav=path/to/noisy.wav
# or:
make enhance IN_WAV=path/to/noisy.wav
```

---

## 📊 Evaluation

```bash
python -m se_cli.cli eval --config configs/exp_mamba_unet.yaml
# or:
make eval
```

Typical metrics: **PESQ**, **STOI**, **SI-SDR**, **SegSNR** (configurable in YAML).

---

## 📦 ONNX Export

```bash
# basic
make export

# or explicit:
python deploy/export_onnx_min.py   --config configs/exp_mamba_unet.yaml   --out artifacts/export/mamba_unet_auto.onnx   --opset 17 --sample_len 16000
```

If the model isn't automatically loaded from `model.py`, specify the builder:

```bash
make export MODEL_SPEC=se_models.mamba_unet.model:build_model
```

After export:

```bash
make onnx-sanity   # validates the model + 1 forward pass in ORT
```

---

## 🔬 PT ↔ ONNX Parity

```bash
make parity MODEL_SPEC=se_models.mamba_unet.model
# internally: tools/parity_onnx.py --T 24000 --tol 0.01
```

Measures signal deviation between PyTorch and ONNX Runtime (default threshold ±1%).

---

## ⚖️ Quantization (Dynamic INT8)

```bash
make quantize
# output: artifacts/export/mamba_unet_auto.int8.onnx
```

Static quantization (calibration-based) can be added later if needed.

---

## 🧪 Tests & CI

Run local tests:

```bash
make test
```

Lint & type checks:

```bash
make lint && make type
```

CI: workflows under `.github/workflows/` run lint, type check, ONNX export, sanity, parity, and quantization on Python 3.10/3.11.

---

## 🗂️ Data / Manifests

Expected CSV format:

```
noisy,clean
/path/to/mix.wav,/path/to/clean.wav
```

Segmentation/sample rate settings are defined via `train.segment_sec` and `data.sample_rate` in YAML (default: 16 kHz).

---

## 🧩 Troubleshooting

**ModuleNotFoundError: se_models...**
→ Ensure you ran `make setup (install -e .)` or set `MODEL_MODULE/model.module` in YAML (e.g., `se_models.mamba_unet.model:build_model`).

**sqlite3 error during pre-commit**
→ Install `pysqlite3-binary` and inject via `sitecustomize.py`, or recreate the venv using a Python build that includes sqlite.

**Large PT↔ONNX differences**
→ Increase `--sample_len` during export, check opset, disable dropout/randomness (fix seed), and rerun `make parity`.
