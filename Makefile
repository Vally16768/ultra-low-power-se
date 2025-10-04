# -----------------------------
# Ultra-Low-Power SE — Makefile
# -----------------------------
.RECIPEPREFIX := >
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

ifneq (,$(wildcard .env.local))
include .env.local
export
endif

# ---------- Config ----------
VENV          ?= .venv
PYTHON        ?= python3
PIP           ?= $(VENV)/bin/pip
ACTIVATE      = . $(VENV)/bin/activate

PWD           := $(shell pwd)
ARTIFACTS     ?= $(PWD)/artifacts
EXPORT_DIR    ?= $(ARTIFACTS)/export

CFG           ?= configs/exp_mamba_unet.yaml

# ---------- Venv ----------
$(VENV)/bin/activate: pyproject.toml requirements.txt
> test -d $(VENV) || $(PYTHON) -m venv $(VENV)
> $(PIP) install -U pip wheel
> if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
> $(PIP) install -e .
> touch $@

.PHONY: setup
setup: $(VENV)/bin/activate
> echo "[ok] venv & deps"

# ---------- Train / Eval ----------
.PHONY: train
train: $(VENV)/bin/activate
> $(ACTIVATE); python -m se_cli.cli train --config $(CFG)

.PHONY: eval
eval: $(VENV)/bin/activate
> $(ACTIVATE); python -m se_cli.cli eval --config $(CFG)

.PHONY: enhance
enhance: $(VENV)/bin/activate
> test -n "$(IN_WAV)" || (echo "Setează IN_WAV=path.wav"; exit 1)
> $(ACTIVATE); python -m se_cli.cli enhance --config $(CFG) -o inference.in_wav=$(IN_WAV)

.PHONY: parity
parity: $(VENV)/bin/activate export
> $(ACTIVATE); python tools/parity_onnx.py \
>   --model se_models.mamba_unet.model:build_model \
>   --config $(CFG) \
>   --onnx artifacts/export/mamba_unet_auto.onnx \
>   --T 24000 --tol 0.01

.PHONY: quantize
quantize: $(VENV)/bin/activate export
> $(ACTIVATE); python deploy/quantize_onnx.py \
>   --in_model artifacts/export/mamba_unet_auto.onnx \
>   --out_model artifacts/export/mamba_unet_auto.int8.onnx \
>   --weight_type QInt8

# ---------- Export ONNX ----------
.PHONY: export
export: $(VENV)/bin/activate
> $(ACTIVATE); python deploy/export_onnx_min.py \
>   --model se_models.mamba_unet.model:build_model \
>   --config $(CFG) \
>   --out $(EXPORT_DIR)/mamba_unet_auto.onnx

.PHONY: onnx-sanity
onnx-sanity: $(VENV)/bin/activate export
> $(ACTIVATE); python - <<'PY'
> import onnx, onnxruntime as ort, numpy as np
> m="artifacts/export/mamba_unet_auto.onnx"
> onnx.checker.check_model(onnx.load(m))
> sess=ort.InferenceSession(m, providers=["CPUExecutionProvider"])
> x=np.zeros((1,1,16000),np.float32)
> y=sess.run(["enhanced"],{"noisy":x})[0]
> assert y.shape[0]==1 and y.shape[1]==1 and y.shape[2]==16000
> print("[onnx] sanity OK:", y.shape)
> PY

# ---------- Clean ----------
.PHONY: clean
clean:
> rm -rf $(ARTIFACTS)/export/*

.PHONY: distclean
distclean: clean
> rm -rf $(VENV)


.PHONY: lint type test
lint: $(VENV)/bin/activate
> $(ACTIVATE); ruff check .

type: $(VENV)/bin/activate
> $(ACTIVATE); mypy .

test: $(VENV)/bin/activate
> $(ACTIVATE); pytest -q

# deja ai export/onnx-sanity; mai sus ai și parity/quantize
