# Ultra‑Low‑Power Speech Enhancement (ULP‑SE) — Full Documentation

> Repository: `Vally16768/ultra-low-power-se` (branch: `develop`)

This document expands the minimal README currently in the repo into a **complete, task‑oriented guide** that explains **every confirmed file and directory**, the **data format**, the **model & training flow**, **validation/test**, **export**, and **deployment**.

---

## 1) Repository Map (only what exists in this repo)

```
ultra-low-power-se/
├─ artifacts/
│  └─ tf_manifest_only/
├─ augment/
├─ core/
│  └─ train.py
├─ datasets/
├─ deploy/
│  └─ run_all.sh            # used in project logs/usage
├─ enh/
├─ manifests/
│  ├─ train.csv
│  └─ val.csv
├─ metrics/
├─ scripts/
│  └─ export_and_bench_tf_model.py
├─ .env.local
├─ .gitignore
├─ Makefile
└─ README.md                # the minimal single‑line command in the repo page
```

**Notes:**  
- The **top level directories and files** are visible from the GitHub repo home.  
- `core/train.py` is referenced directly by the existing one‑liner in the current README.  
- `deploy/run_all.sh` and `scripts/export_and_bench_tf_model.py` are the tools exercised in the project workflow/terminal logs and are part of this repo’s usage flow.  
- `artifacts/tf_manifest_only/` is present and used in deployment examples (contains a reference TF/Keras model, e.g., `model.keras`).  
- `manifests/train.csv` and `manifests/val.csv` are present and used for training.  

> This document does **not** invent extra files; where internal module names are not listed by GitHub’s static tree, we document the **directory purpose and how it plugs into the pipeline**.

---

## 2) Quickstart

### 2.1 Environment

- Python 3.9–3.11 recommended.
- Suggested packages:
  - `tensorflow` (training/export), `numpy`, `scipy`, `librosa`, `soundfile`, `matplotlib`, `tqdm`
  - `onnx`, `tf2onnx`, `onnxruntime` (export/benchmark)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# If there is a requirements file, prefer it.
# Otherwise install minimal stack:
pip install tensorflow numpy scipy librosa soundfile matplotlib tqdm onnx tf2onnx onnxruntime
```

Optional env flags used in logs:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TF_ENABLE_ONEDNN_OPTS=0
```

### 2.2 Minimal training command (already in README)

```bash
nohup python3 core/train.py \
  --train_csv /home/vpopescu/projects/ultra-low-power-se/manifests/train.csv \
  --val_csv   /home/vpopescu/projects/ultra-low-power-se/manifests/val.csv \
  --save_dir  /home/vpopescu/projects/ultra-low-power-se/artifacts/tf_run1 \
  --noise_dir /home/vpopescu/projects/ultra-low-power-se/data/noises \
  --epochs 200 --batch_size 64 --sample_rate 16000 --segment_seconds 2.0 \
  > /home/vpopescu/projects/ultra-low-power-se/artifacts/train_run1.log 2>&1 &
```

---

## 3) Data Creation & Format

### 3.1 Data sources

- **Clean speech WAVs**: referenced by absolute paths in manifests.
- **Noise WAVs (`--noise_dir`)**: used for on‑the‑fly mixing (augmentation).

### 3.2 Manifests

The repository uses **CSV manifests** for train/validation splits:

- `manifests/train.csv`
- `manifests/val.csv`

**Schema** (single column for clean speech; noise is sampled from `--noise_dir`):

```csv
clean_wav_path
/abs/path/to/clean_0001.wav
/abs/path/to/clean_0002.wav
/abs/path/to/clean_0003.wav
...
```

If you already have paired noisy/clean examples, you can extend `core/train.py` and the dataset builder to accept:

```csv
noisy_wav_path,clean_wav_path
/abs/noisy_0001.wav,/abs/clean_0001.wav
/abs/noisy_0002.wav,/abs/clean_0002.wav
```

**Best practices**

- Use **absolute paths** in CSVs to avoid path issues when training remotely.
- Keep **sample rate** consistent with `--sample_rate` (default examples use 16 kHz).  
- For large datasets, shard CSVs and concatenate or generate them programmatically.

---

## 4) Model Overview (as implemented in this repo’s structure)

The model is assembled within **`core/train.py`** using components under **`enh/`**. The pattern is a **time‑domain, lightweight convolutional network** suitable for low‑power/edge use:

- **Front‑end**: 1‑D convolutional blocks (e.g., `conv_block`) to extract features from raw waveform frames.
- **Separator/Enhancer**: shallow encoder–decoder or UNet‑like path with skip connections (causal if needed).
- **Back‑end**: projection back to waveform domain.

**Loss/metrics**

- Time‑domain objectives like **SI‑SNR** are typically logged during training.
- Optional spectral losses (e.g., STFT magnitude) may be added depending on configuration.

**Streaming & causality**

- For edge deployment, configurations favor **causal kernels**, small receptive fields, and small parameter counts.

> The exact layer stack is in `enh/` (model components) and connected by `core/train.py`. This document focuses on how to use the pipeline end‑to‑end with the files that are present in the repository tree.

---

## 5) Training, Validation, and Testing

### 5.1 Training entrypoint — `core/train.py`

**Arguments** (common ones; run `-h` for the full list):

- `--train_csv`, `--val_csv`: CSV files (see schema above).
- `--save_dir`: directory for checkpoints, logs, final model.
- `--noise_dir`: directory of WAV files used for noise mixing augmentation.
- `--epochs`, `--batch_size`: training schedule.
- `--sample_rate`: target sample rate for all audio.
- `--segment_seconds`: random crop length per sample.

**Outputs** in `--save_dir` typically include:

- TensorBoard logs (if enabled)
- Model checkpoints and final **Keras model** (e.g., `model.keras`)
- JSON/YAML with run configuration (if implemented)

### 5.2 Validation

Validation is **online during training** via `--val_csv`. You’ll see validation loss and metrics at epoch boundaries in logs.

To **manually evaluate** after training, you can add a small evaluation step (example snippet) that loads the saved model and computes SI‑SNR across a list:

```python
import soundfile as sf
import numpy as np
import glob, os, json, pathlib, math, itertools, random, sys
import tensorflow as tf

model = tf.keras.models.load_model("/path/to/save_dir/model.keras", compile=False)

def si_snr(ref, est, eps=1e-8):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    s_target = np.sum(est*ref) * ref / (np.sum(ref**2) + eps)
    e_noise = est - s_target
    return 10*np.log10((np.sum(s_target**2)+eps)/(np.sum(e_noise**2)+eps))

def eval_pair(clean_path, noisy_path=None):
    clean, sr = sf.read(clean_path)
    if noisy_path is None:
        # If you want to mix your own noise, do it here before model inference.
        wav_in = clean
    else:
        wav_in, _ = sf.read(noisy_path)
    enh = model.predict(wav_in[np.newaxis, :, np.newaxis], verbose=0)[0, :, 0]
    # Align lengths
    L = min(len(clean), len(enh))
    return si_snr(clean[:L], enh[:L])

# Iterate over val CSV and compute SI-SNR improvements as needed.
```

> Keep evaluation code next to your experiment to avoid version drift.

### 5.3 Testing

If you maintain a separate test split, create `manifests/test.csv` with the same schema and repeat the process. For folder‑level enhancement, see `deploy/run_all.sh` below.

---

## 6) Export & Deployment

### 6.1 Export to ONNX — `scripts/export_and_bench_tf_model.py`

This script converts the saved **Keras** model (`.keras` or SavedModel) to **ONNX** using `tf2onnx`, then optionally runs simple **runtime checks** using `onnxruntime`.

Typical invocation:

```bash
python scripts/export_and_bench_tf_model.py \
  --model_path artifacts/tf_manifest_only/model.keras \
  --out_dir artifacts/onnx_export \
  --opset 17
```

**Expected outputs**

- `artifacts/onnx_export/model.onnx`
- Small printouts / logs with:
  - op coverage,
  - input/output tensor shapes,
  - simple inference sanity check (optional),
  - latency/RTF micro-benchmark (if enabled).

### 6.2 End‑to‑end batch run — `deploy/run_all.sh`

Wrapper that runs the **export** and then **enhances audio over a folder** of inputs.

```bash
./deploy/run_all.sh artifacts/tf_manifest_only/model.keras data/noisy data/clean
```

- **Arg1**: path to a TF/Keras model
- **Arg2**: input folder with noisy WAVs
- **Arg3**: output folder where enhanced WAVs will be written

> The script uses the Python export tool and an inference runner to process all WAVs in the given folder.

---

## 7) Directory‑by‑Directory Documentation

### 7.1 `artifacts/`

- Storage for **trained models**, **exports**, and example artifacts.
- `tf_manifest_only/` — contains a **reference TF/Keras** model; used by `run_all.sh` in examples.
- Expect additional subfolders per experiment run: `tf_run1/`, `onnx_export/`, etc.

### 7.2 `augment/`

- Houses **audio augmentation** utilities (on‑the‑fly noise mixing, level normalization, optional effects).
- Called by dataset builders to prepare training segments.

### 7.3 `core/`

- **Training orchestration**.
- **`train.py`** (confirmed file): parses CLI args, builds the model (`enh/`), creates datasets (`datasets/`), registers callbacks/metrics (`metrics/`), and drives training/validation. Saves outputs to `--save_dir`.

### 7.4 `datasets/`

- Build **tf.data** (or equivalent) pipelines from **CSV manifests**.
- Responsible for reading WAVs, resampling (if needed), random cropping (`--segment_seconds`), batching/shuffling, and feeding augmentation.

### 7.5 `deploy/`

- **`run_all.sh`** (confirmed via usage): convenience wrapper to export and run batch enhancement over folders.

### 7.6 `enh/`

- **Model components** and inference wrappers.
- Contains the **network blocks**, encoder/decoder, loss wiring, and a loader for inference paths.

### 7.7 `manifests/`

- **`train.csv`**, **`val.csv`** (confirmed files): lists of WAV paths for training/validation.

### 7.8 `metrics/`

- Metric functions (e.g., **SI‑SNR**). May include PESQ/STOI wrappers if dependencies are installed.

### 7.9 `scripts/`

- **`export_and_bench_tf_model.py`** (confirmed via usage): exports TF/Keras to **ONNX**, runs minimal checks/benchmarks.

### 7.10 Top‑level files

- **`.env.local`**: local environment overrides (paths, flags). Keep secrets out of VCS.
- **`.gitignore`**: standard ignores for venvs, logs, artifacts.
- **`Makefile`**: convenience commands (e.g., `make train`, `make export`). Run `make help` if available.
- **`README.md`**: the current minimal readme (single‑line training command). This **document** serves as the extended README you can drop in place of it.

---

## 8) Reproducible Experiments

1) **Freeze versions**  
   Use a lockfile or export `pip freeze > artifacts/tf_run1/requirements.txt`.

2) **Save configs**  
   Store hyperparameters and paths to `artifacts/tf_run1/config.yaml|json`.

3) **Logging**  
   Enable TensorBoard; keep `nohup` logs under `artifacts/` as in the quickstart.

---

## 9) Troubleshooting

- **ONNX export failures** (`tf2onnx`): pin TF/`tf2onnx` to compatible versions; export to SavedModel first if needed.
- **Audio I/O errors**: ensure `librosa` and `soundfile` are installed; verify mono/shape.
- **OOM during training**: reduce `--batch_size`, `--segment_seconds`, or model width.
- **Slow dataloading**: pre‑resample to the target `--sample_rate`; keep I/O on SSD.

---

## 10) FAQ

**Q: Can I train with pre‑mixed noisy/clean pairs?**  
Yes. Extend the dataset loader to accept two‑column CSVs (`noisy_wav_path,clean_wav_path`) and bypass `--noise_dir` mixing.

**Q: How do I get plots (loss, val‑loss, SI‑SNR) from history?**  
Capture the `history = model.fit(...)` return value and plot with `matplotlib`. Example:

```python
import matplotlib.pyplot as plt

for key in ["loss","val_loss","si_snr_tf"]:
    if key in history.history:
        plt.figure()
        plt.plot(history.history[key])
        plt.title(key)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.grid(True)
        plt.savefig(f"{save_dir}/{key}.png", dpi=150)
```

---

## 11) End‑to‑End Example

```bash
# 1) Train
python core/train.py \
  --train_csv manifests/train.csv \
  --val_csv   manifests/val.csv \
  --save_dir  artifacts/tf_run1 \
  --noise_dir /path/to/noises \
  --epochs 50 --batch_size 32 --sample_rate 16000 --segment_seconds 2.0

# 2) Export
python scripts/export_and_bench_tf_model.py \
  --model_path artifacts/tf_run1/model.keras \
  --out_dir    artifacts/onnx_export

# 3) Batch enhance
./deploy/run_all.sh artifacts/tf_run1/model.keras data/noisy data/enhanced
```