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

# Dataset roots (override în .env.local la nevoie)
PWD           := $(shell pwd)
LIBRISPEECH_ROOT ?= $(PWD)/data/librispeech/LibriSpeech
DEMAND_ROOT      ?= $(PWD)/data/noise/demand
RIRS_ROOT        ?= $(PWD)/data/rirs
VBD_ROOT         ?= $(PWD)/data/voicebank-demand-16k
MIXGEN           ?= datasets/mixgen.py

# Fișiere/dirs
REQ_FILE     ?= requirements.txt
CFG_TRAIN    ?= configs/exp_mamba_unet.yaml
ONNX_PATH    ?= artifacts/export/mamba_unet_auto.onnx
DATA_FINAL_DIR ?= data/datasets/final
UNIFIED_SCRIPT ?= scripts/build_unified_dataset.py

# Model module (override în .env.local dacă vrei alt model)
MODEL_MODULE ?= se_models.mamba_unet.model

# ---------- Phony ----------
.PHONY: help setup setup-dev ensure-venv \
        test lint fmt \
        data-train data-verify datasets datasets.verify datasets.clean \
        dataset-unified dataset-unified.clean \
        train tb logs export onnx onnx-verify onnx-verify-sim onnx-sanity \
        infer eval print-vars check-scripts \
        clean clean-all clean-tb

# ---------- Help ----------
help:
> @echo "Targets:"
> @echo "  setup               - create venv & install deps (requirements.txt)"
> @echo "  setup-dev           - install extra dev deps (ruff, mypy, pytest)"
> @echo "  test                - run unit tests"
> @echo "  lint                - ruff + mypy (non-strict)"
> @echo "  fmt                 - ruff --fix"
> @echo "  data-train          - build VoiceBank-DEMAND train/test manifests (detect *_wav)"
> @echo "  data-verify         - integrity check & manifests for VoiceBank/LibriSpeech"
> @echo "  datasets            - FULL pipeline (prepare dirs/lists & generate mixes)"
> @echo "  datasets.verify     - verify generated datasets integrity"
> @echo "  datasets.clean      - remove only generated mixes (data/prepared/*)"
> @echo "  dataset-unified     - build unified train/val (VBD ∪ Libri+noise augment)"
> @echo "  dataset-unified.clean - remove unified dataset folder"
> @echo "  train               - run training via se_cli (cfg YAML controlează tot)"
> @echo "  tb                  - launch TensorBoard on artifacts"
> @echo "  logs                - tail -f training log (stdout mirror)"
> @echo "  export              - export ONNX (static input=noisy -> [B,1,T])"
> @echo "  onnx                - alias pt. onnx-verify"
> @echo "  onnx-verify         - verify ONNX vs PyTorch outputs"
> @echo "  onnx-verify-sim     - verify ONNX (sim)"
> @echo "  onnx-sanity         - parity Torch vs ONNX (script extra)"
> @echo "  print-vars          - show key env/paths used by the pipeline"
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
> $(ACTIVATE) && mypy --ignore-missing-imports se_cli runners deploy datasets se_models || true

fmt:
> $(ACTIVATE) && ruff check --fix .

# ---------- Data (VoiceBank-DEMAND pairs) ----------
data-train: check-scripts
> VBD_ROOT="$(VBD_ROOT)" bash scripts/build_vbd_pairs.sh

# verificări rapide VoiceBank + LibriSpeech (manifest + număr fișiere, rate etc.)
data-verify: check-scripts
> bash scripts/verify_and_prepare.sh

# ---------- Datasets Orchestration (clasic: pregătește diruri/liste, generează mixuri) ----------
datasets: check-scripts
> LIBRISPEECH_ROOT="$(LIBRISPEECH_ROOT)" \
> DEMAND_ROOT="$(DEMAND_ROOT)" \
> RIRS_ROOT="$(RIRS_ROOT)" \
> MIXGEN="$(MIXGEN)" \
> bash scripts/make_dataset.sh \
>   --librispeech "$(LIBRISPEECH_ROOT)" \
>   --demand "$(DEMAND_ROOT)" \
>   --rirs "$(RIRS_ROOT)"

datasets.verify: check-scripts
> bash scripts/make_dataset.sh --verify-only

datasets.clean:
> rm -rf data/prepared

# ---------- Unified dataset (VBD ∪ Libri augment) ----------
dataset-unified:
> test -f "$(UNIFIED_SCRIPT)" || { echo "[ERR] $(UNIFIED_SCRIPT) missing. Add it, apoi rulează din nou."; exit 2; }
> $(ACTIVATE) && python "$(UNIFIED_SCRIPT)" \
>   --proj_dir "$(PWD)" --mix_per_clean 1 --snr 0,5,10,15

dataset-unified.clean:
> rm -rf "$(DATA_FINAL_DIR)"

# ---------- Train ----------
train:
> mkdir -p artifacts/logs
> $(ACTIVATE) && MODEL_MODULE="$(MODEL_MODULE)" \
> python -m se_cli.cli train --config $(CFG_TRAIN) 2>&1 | tee artifacts/logs/train_$$.log

# ---------- TensorBoard & logs ----------
tb:
> $(ACTIVATE) && tensorboard --logdir artifacts/exp --port 6006

logs:
> tail -f artifacts/logs/*.log

# ---------- Export ----------
export:
> $(ACTIVATE) && python deploy/export_onnx_min.py \
>   --model $(MODEL_MODULE):build_model \
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

# ---------- Hooks opționale ----------
infer:
> $(ACTIVATE) && python -m se_cli.cli infer --config $(CFG_TRAIN)

eval:
> $(ACTIVATE) && python -m se_cli.cli eval --config $(CFG_TRAIN)

# ---------- Utils ----------
print-vars:
> @echo "PWD               : $(PWD)"
> @echo "VENV              : $(VENV)"
> @echo "MODEL_MODULE      : $(MODEL_MODULE)"
> @echo "CFG_TRAIN         : $(CFG_TRAIN)"
> @echo "ONNX_PATH         : $(ONNX_PATH)"
> @echo "VBD_ROOT          : $(VBD_ROOT)"
> @echo "LIBRISPEECH_ROOT  : $(LIBRISPEECH_ROOT)"
> @echo "DEMAND_ROOT       : $(DEMAND_ROOT)"
> @echo "RIRS_ROOT         : $(RIRS_ROOT)"
> @echo "MIXGEN            : $(MIXGEN)"
> @echo "UNIFIED_SCRIPT    : $(UNIFIED_SCRIPT)"
> @echo "DATA_FINAL_DIR    : $(DATA_FINAL_DIR)"

check-scripts:
> test -f scripts/build_vbd_pairs.sh || { echo "[ERR] scripts/build_vbd_pairs.sh missing"; exit 2; }
> test -f scripts/make_dataset.sh    || { echo "[ERR] scripts/make_dataset.sh missing"; exit 2; }

# ---------- Cleanup ----------
clean:
> scripts/cleanup.sh --yes

clean-all:
> scripts/cleanup.sh --aggressive --yes

clean-tb:
> scripts/cleanup.sh --only tb --yes
