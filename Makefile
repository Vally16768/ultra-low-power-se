# -----------------------------
# Ultra-Low-Power SE — Makefile
# -----------------------------
# Folosim '>' în loc de TAB pentru liniile de comandă (evită "missing separator")
.RECIPEPREFIX := >
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

# Dacă există .env.local, îl încărcăm (ex: LIBRISPEECH_ROOT=/data/LibriSpeech etc.)
ifneq (,$(wildcard .env.local))
include .env.local
export
endif

# ---------- Config ----------
VENV         ?= .venv
PYTHON       ?= python3
PIP          ?= $(VENV)/bin/pip
ACTIVATE     = . $(VENV)/bin/activate

# Rute dataset (pot fi suprascrise din .env.local)
LIBRISPEECH_ROOT ?= /data/LibriSpeech
DEMAND_ROOT      ?= /data/DEMAND
RIRS_ROOT        ?= /data/RIRS_NOISES

# Fişiere/dirs utilizate de ținte
REQ_FILE   ?= requirements.txt
CFG_TRAIN  ?= configs/exp_mamba_unet.yaml
ONNX_PATH  ?= artifacts/export/mamba_unet_auto.onnx

# ---------- Phony ----------
.PHONY: help setup setup-dev ensure-venv \
        test lint fmt \
        data-train datasets datasets.verify datasets.clean \
        train export onnx onnx-verify onnx-verify-sim onnx-sanity \
        infer eval \
        clean clean-all clean-tb

# ---------- Help ----------
help:
> @echo "Targets:"
> @echo "  setup              - create venv & install deps (requirements.txt)"
> @echo "  setup-dev          - install extra dev deps (ruff, mypy, pytest)"
> @echo "  test               - run unit tests"
> @echo "  lint               - ruff + mypy (non-strict)"
> @echo "  fmt                - ruff --fix"
> @echo "  data-train         - prepare VoiceBank+DEMAND pairs (VBD manifest)"
> @echo "  datasets           - FULL pipeline: prepare dirs/lists & generate mixes (idempotent)"
> @echo "  datasets.verify    - verify generated datasets integrity"
> @echo "  datasets.clean     - remove only generated mixes (data/prepared/*)"
> @echo "  train              - run training via se_cli"
> @echo "  export             - export ONNX (robust, static, input=noisy, out=[B,1,T])"
> @echo "  onnx               - alias pt. onnx-verify"
> @echo "  onnx-verify        - verify ONNX vs PyTorch outputs"
> @echo "  onnx-verify-sim    - verify ONNX (sim)"
> @echo "  onnx-sanity        - parity Torch vs ONNX (script extra)"
> @echo "  clean / clean-all / clean-tb"

# ---------- Env / Deps ----------
ensure-venv:
> if [ ! -d "$(VENV)" ]; then \
>   $(PYTHON) -m venv $(VENV); \
> fi

setup: ensure-venv
> $(ACTIVATE) && python -m pip install --upgrade pip
> $(ACTIVATE) && pip install -r $(REQ_FILE)

setup-dev: ensure-venv
> $(ACTIVATE) && pip install -U pip ruff mypy pytest

# ---------- QA ----------
test:
> $(ACTIVATE) && pytest -q

lint:
> $(ACTIVATE) && ruff check .
> $(ACTIVATE) && mypy --ignore-missing-imports se_cli runners deploy datasets se_models

fmt:
> $(ACTIVATE) && ruff check --fix .

# ---------- Data (VoiceBank-DEMAND pairs) ----------
data-train:
> bash scripts/build_vbd_pairs.sh

# ---------- Datasets Orchestration ----------
datasets:
> bash scripts/make_dataset.sh \
>   --librispeech "$(LIBRISPEECH_ROOT)" \
>   --demand "$(DEMAND_ROOT)" \
>   --rirs "$(RIRS_ROOT)"

datasets.verify:
> bash scripts/make_dataset.sh --verify-only

datasets.clean:
> rm -rf data/prepared

# ---------- Train / Export / Eval ----------
train:
> $(ACTIVATE) && python -m se_cli.cli train --config $(CFG_TRAIN)

export:
> $(ACTIVATE) && python deploy/export_onnx_min.py \
>   --model se_models.mamba_unet.model:build_model \
>   --config $(CFG_TRAIN) \
>   --out $(ONNX_PATH) \
>   --opset 17 --dynamic 0 \
>   --input-name noisy \
>   --keep-ch 1

# ---------- ONNX Verify ----------
onnx: onnx-verify

onnx-verify:
> $(ACTIVATE) && python deploy/verify_onnx.py $(ONNX_PATH) --config $(CFG_TRAIN)

onnx-verify-sim:
> $(ACTIVATE) && python deploy/verify_onnx.py $(ONNX_PATH:.onnx=.sim.onnx) --config $(CFG_TRAIN)

onnx-sanity:
> $(ACTIVATE) && python deploy/sanity_onnx.py $(ONNX_PATH)

# hooks optionale (dacă ai CLI pentru infer/eval)
infer:
> $(ACTIVATE) && python -m se_cli.cli infer --config $(CFG_TRAIN)

eval:
> $(ACTIVATE) && python -m se_cli.cli eval --config $(CFG_TRAIN)

# ---------- Cleanup ----------
clean:
> scripts/cleanup.sh --yes

clean-all:
> scripts/cleanup.sh --aggressive --yes

clean-tb:
> scripts/cleanup.sh --only tb --yes
