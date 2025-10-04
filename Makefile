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
# Poți suprascrie MODEL_SPEC când vrei alt builder, ex:
# make export MODEL_SPEC=se_models.mamba_unet.model:build_model
MODEL_SPEC    ?=
OPSET         ?= 17
SAMPLE_LEN    ?= 16000

# ---------- Venv ----------
$(VENV)/bin/activate: pyproject.toml
> test -d $(VENV) || $(PYTHON) -m venv $(VENV)
> $(PIP) install -U pip wheel
> $(PIP) install -e .
> touch $@

.PHONY: setup
setup: $(VENV)/bin/activate
> echo "[ok] venv & deps (editable install)"

.PHONY: dev-setup
dev-setup: $(VENV)/bin/activate
> $(ACTIVATE); pip install -U ruff mypy pytest onnx onnxruntime pre-commit
> echo "[ok] dev deps (ruff/mypy/pytest/onnx/onnxruntime)"

# ---------- Train / Eval / Enhance ----------
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

# ---------- Export ONNX ----------
.PHONY: export
export: $(VENV)/bin/activate
> mkdir -p $(EXPORT_DIR)
> if [ -n "$(MODEL_SPEC)" ]; then \
>   $(ACTIVATE); python deploy/export_onnx_min.py \
>     --model $(MODEL_SPEC) \
>     --config $(CFG) \
>     --out $(EXPORT_DIR)/mamba_unet_auto.onnx \
>     --opset $(OPSET) \
>     --sample_len $(SAMPLE_LEN); \
> else \
>   $(ACTIVATE); python deploy/export_onnx_min.py \
>     --config $(CFG) \
>     --out $(EXPORT_DIR)/mamba_unet_auto.onnx \
>     --opset $(OPSET) \
>     --sample_len $(SAMPLE_LEN); \
> fi

.PHONY: onnx-sanity
onnx-sanity: $(VENV)/bin/activate export
> $(ACTIVATE); python - <<'PY'
> import onnx, onnxruntime as ort, numpy as np, os
> m=os.path.join("artifacts","export","mamba_unet_auto.onnx")
> onnx.checker.check_model(onnx.load(m))
> s=ort.InferenceSession(m, providers=["CPUExecutionProvider"])
> x=np.zeros((1,1,16000),np.float32)
> y=s.run(["enhanced"],{"noisy":x})[0]
> assert y.shape==(1,1,16000)
> print("[onnx] sanity OK:", y.shape)
> PY

# ---------- Parity PT ↔ ONNX ----------
.PHONY: parity
parity: $(VENV)/bin/activate export
> $(ACTIVATE); python tools/parity_onnx.py \
>   --model $${MODEL_SPEC:-se_models.mamba_unet.model} \
>   --config $(CFG) \
>   --onnx $(EXPORT_DIR)/mamba_unet_auto.onnx \
>   --T 24000 --tol 0.01

# ---------- Quantize ----------
.PHONY: quantize
quantize: $(VENV)/bin/activate export
> $(ACTIVATE); python deploy/quantize_onnx.py \
>   --in_model $(EXPORT_DIR)/mamba_unet_auto.onnx \
>   --out_model $(EXPORT_DIR)/mamba_unet_auto.int8.onnx \
>   --weight_type QInt8

# ---------- Quality (lint / type / test) ----------
.PHONY: lint
lint: dev-setup
> $(ACTIVATE); ruff check .

.PHONY: type
type: dev-setup
> $(ACTIVATE); mypy .

.PHONY: test
test: dev-setup
> $(ACTIVATE); if [ -d tests ]; then pytest -q; else echo "[tests] no tests/ directory — skip"; fi

# ---------- Clean ----------
.PHONY: clean
clean:
> rm -rf $(EXPORT_DIR)/*

.PHONY: distclean
distclean: clean
> rm -rf $(VENV)

# ---------- Pre-commit ----------
.PHONY: pre-commit-install pre-commit-update pre-commit
pre-commit-install: dev-setup
> $(ACTIVATE); pre-commit install -t pre-commit -t commit-msg -t pre-push
> echo "[pre-commit] hooks instalate (pre-commit / commit-msg / pre-push)"

pre-commit-update: dev-setup
> $(ACTIVATE); pre-commit autoupdate
> echo "[pre-commit] hooks actualizate"

# rulează toate hook-urile pe tot repo-ul (ideal înainte de primul commit mare)
pre-commit: dev-setup
> $(ACTIVATE); pre-commit run --all-files
