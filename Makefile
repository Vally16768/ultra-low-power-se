# -----------------------------
# Ultra-Low-Power SE — Makefile
# -----------------------------

.PHONY: help setup train eval enhance export score onnx-sanity \
        datasets datasets-only-manifests mix-train mix-dev vbd-test \
        clean deepclean

# ---- Virtualenv & tools ----
VENV    ?= .venv
PIP     := $(VENV)/bin/pip
PYTHON  := $(VENV)/bin/python

# ---- Experiment config ----
CFG         ?= configs/exp_robustnet_plus.yaml
ONNX_MODEL  ?= artifacts/export/robustnet_plus.onnx

# ---- Data paths (override in .env.local if you like) ----
DATA_ROOT       ?= data
PREPARED_ROOT   ?= $(DATA_ROOT)/prepared
TRAIN_MIXES_DIR ?= $(PREPARED_ROOT)/train_mixes
DEV_MIXES_DIR   ?= $(PREPARED_ROOT)/dev_mixes
VBD_TEST_DIR    ?= $(PREPARED_ROOT)/voicebank/test

# Canonical manifests
TRAIN_CSV ?= $(TRAIN_MIXES_DIR)/manifests/pairs.csv
VAL_CSV   ?= $(DEV_MIXES_DIR)/manifests/pairs.csv
TEST_CSV  ?= $(VBD_TEST_DIR)/manifests/pairs.csv

# -----------------------------
# Top-level helpers
# -----------------------------

help:
	@echo ""
	@echo "Targets:"
	@echo "  setup            Create venv and install deps"
	@echo "  train            Train model using $(CFG)"
	@echo "  eval             Run evaluation (PESQ/STOI/etc.)"
	@echo "  enhance          Enhance one wav: make enhance IN_WAV=... OUT_WAV=..."
	@echo "  export           Export to ONNX"
	@echo "  onnx-sanity      Quick ORT sanity check on exported ONNX"
	@echo "  score            Score already-enhanced wavs"
	@echo "  datasets         Create mixes + test manifests if missing"
	@echo "  clean|deepclean  Remove artifacts / venv"
	@echo ""

# -----------------------------
# Env & Python deps
# -----------------------------

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip wheel setuptools
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install pyyaml tensorboard tqdm soundfile numpy pesq pystoi onnx onnxruntime

# -----------------------------
# CLI entrypoints
# -----------------------------

train:
	$(PYTHON) -m se_cli.cli train --config $(CFG)

eval:
	$(PYTHON) -m se_cli.cli eval --config $(CFG)

enhance:
	@test -n "$(IN_WAV)" -a -n "$(OUT_WAV)" || (echo "Usage: make enhance IN_WAV=<in.wav> OUT_WAV=<out.wav>"; exit 2)
	$(PYTHON) -m se_cli.cli enhance --config $(CFG) \
		-o inference.in_wav=$(IN_WAV) -o inference.out_wav=$(OUT_WAV)

export:
	$(PYTHON) -m se_cli.cli export --config $(CFG)

score:
	$(PYTHON) -m se_cli.cli score --config $(CFG)

# Do a tiny forward pass via ONNX Runtime to check shapes/wiring.
# Depends on 'export' so the model exists.
onnx-sanity: export
	$(PYTHON) - <<'PY'
import onnx, onnxruntime as ort, numpy as np, sys
m = "$(ONNX_MODEL)"
onnx.checker.check_model(m)
sess = ort.InferenceSession(m, providers=['CPUExecutionProvider'])
# Adjust input name if your exported model uses a different key.
input_name = sess.get_inputs()[0].name
x = np.random.randn(1, 1, 16000).astype(np.float32)
y = sess.run(None, {input_name: x})[0]
print("[onnx-sanity]", m, "->", y.shape)
PY

# -----------------------------
# Datasets (mix generation + manifests)
# -----------------------------

datasets: mix-train mix-dev vbd-test datasets-only-manifests
	@echo "[datasets] done."

# Only (re)build manifest CSVs from what's already on disk.
datasets-only-manifests:
	$(PYTHON) tools/build_manifests.py \
		--train-root $(TRAIN_MIXES_DIR) \
		--val-root   $(DEV_MIXES_DIR) \
		--test-root  $(VBD_TEST_DIR)

# Generate training mixes if folder missing or empty.
mix-train:
	@if [ ! -d "$(TRAIN_MIXES_DIR)" ] || [ -z "$$(ls -A "$(TRAIN_MIXES_DIR)" 2>/dev/null)" ]; then \
	  $(PYTHON) tools/make_mixes.py \
	    --clean-list data/lists/train_clean.txt \
	    --noise-list data/lists/noise_train.txt \
	    --out-root $(TRAIN_MIXES_DIR) --pairs 120000 --snr "0,5,10,15"; \
	else echo "[mix-train] reuse: $(TRAIN_MIXES_DIR)"; fi

# Generate dev/validation mixes if folder missing or empty.
mix-dev:
	@if [ ! -d "$(DEV_MIXES_DIR)" ] || [ -z "$$(ls -A "$(DEV_MIXES_DIR)" 2>/dev/null)" ]; then \
	  $(PYTHON) tools/make_mixes.py \
	    --clean-list data/lists/dev_clean.txt \
	    --noise-list data/lists/noise_dev.txt \
	    --out-root $(DEV_MIXES_DIR) --pairs 8000 --snr "0,5,10,15"; \
	else echo "[mix-dev] reuse: $(DEV_MIXES_DIR)"; fi

# Ensure the canonical VoiceBank-DEMAND test manifest exists.
vbd-test:
	@if [ ! -f "$(TEST_CSV)" ]; then \
	  $(PYTHON) tools/build_manifests.py --test-root $(VBD_TEST_DIR); \
	else echo "[vbd-test] found: $(TEST_CSV)"; fi

# -----------------------------
# Cleanup
# -----------------------------

clean:
	rm -rf artifacts/eval/*

deepclean: clean
	rm -rf $(VENV)
