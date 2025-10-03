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

# ---- Normalize (elimină spații nedorite din .env.local) ----
LIBRISPEECH_ROOT := $(strip $(LIBRISPEECH_ROOT))
DEMAND_ROOT      := $(strip $(DEMAND_ROOT))
RIRS_ROOT        := $(strip $(RIRS_ROOT))
VBD_ROOT         := $(strip $(VBD_ROOT))
MIXGEN           := $(strip $(MIXGEN))

# Fișiere/dirs
REQ_FILE       ?= requirements.txt
CFG_TRAIN      ?= configs/exp_mamba_unet.yaml
ONNX_PATH      ?= artifacts/export/mamba_unet_auto.onnx
DATA_FINAL_DIR ?= data/datasets/final
UNIFIED_SCRIPT ?= scripts/build_unified_dataset.py

# Model module (override în .env.local dacă vrei alt model)
MODEL_MODULE ?= se_models.mamba_unet.model

# ---------- Dirs Lists & Stamps ----------
LISTS_DIR      := lists
CLEAN_LIST     := $(LISTS_DIR)/clean.txt
NOISE_LIST     := $(LISTS_DIR)/noise.txt
RIRS_LIST      := $(LISTS_DIR)/rirs.txt

DATA_DIR       := $(PWD)/data
PREPARED_DIR   := $(DATA_DIR)/prepared
MIX_DIR        := $(PREPARED_DIR)/mix
MIX_TRAIN_DIR  := $(MIX_DIR)/train
MIX_VAL_DIR    := $(MIX_DIR)/val

STAMP_DIR      := $(PREPARED_DIR)/.stamps
$(shell mkdir -p $(STAMP_DIR) >/dev/null)

LIBRISPEECH_STAMP := $(STAMP_DIR)/librispeech.ok
DEMAND_STAMP      := $(STAMP_DIR)/demand.ok
RIRS_STAMP        := $(STAMP_DIR)/rirs.ok
VBD_STAMP         := $(STAMP_DIR)/voicebank.ok
LISTS_STAMP       := $(STAMP_DIR)/lists.ok
MIX_TRAIN_STAMP   := $(STAMP_DIR)/mix_train.ok
MIX_VAL_STAMP     := $(STAMP_DIR)/mix_val.ok

# ---------- Params pentru mixgen ----------
SR            ?= 16000
SNR           ?= -5 0 5 10
SEG_MIN       ?= 2.0
SEG_MAX       ?= 8.0
MIX_SEED      ?= 1337
N_MIX_TRAIN   ?= 5000
N_MIX_VAL     ?= 800
REVERB_PROB   ?= 0.2
CODEC         ?= none      # none|opus_16|opus_24
CLIPPING      ?= none      # none|hard|soft

# ---------- Phony ----------
.PHONY: help setup setup-dev ensure-venv deps-audio \
        test lint fmt \
        data-train data-verify datasets datasets.verify datasets.clean \
        lists lists.clean \
        noise.ensure noise.verify \
        dataset-unified dataset-unified.clean \
        train tb logs export onnx onnx-verify onnx-verify-sim onnx-sanity \
        infer eval print-vars check-scripts \
        clean clean-all clean-tb

# ---------- Help ----------
help:
> @echo "Targets:"
> @echo "  setup               - create venv & install deps (requirements.txt)"
> @echo "  setup-dev           - install extra dev deps (ruff, mypy, pytest)"
> @echo "  deps-audio          - install numpy, soundfile, resampy (în venv)"
> @echo "  test                - run unit tests"
> @echo "  lint                - ruff + mypy (non-strict)"
> @echo "  fmt                 - ruff --fix"
> @echo "  data-train          - build VoiceBank-DEMAND train/test manifests (detect *_wav)"
> @echo "  data-verify         - integrity check & manifests for VoiceBank/LibriSpeech"
> @echo "  datasets            - FULL pipeline (ensure sources, lists, mixes idempotent)"
> @echo "  datasets.verify     - verify generated datasets integrity"
> @echo "  datasets.clean      - remove only generated mixes (data/prepared/*)"
> @echo "  lists               - (re)build clean/noise/rir file lists from roots"
> @echo "  noise.ensure        - create/populate data/noise (link DEMAND or synth fallback)"
> @echo "  noise.verify        - sanity check on data/noise (sr/ch/dur/corruption)"
> @echo "  dataset-unified     - AGREGĂ VBD + prepared mixes (NU generează)"
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
> test -f $(REQ_FILE) && $(ACTIVATE) && pip install -r $(REQ_FILE) || true

setup-dev: ensure-venv
> $(ACTIVATE) && pip install -U pip ruff mypy pytest

# deps audio necesare de pipeline-ul datasets
deps-audio: ensure-venv
> $(ACTIVATE) && pip install -U numpy soundfile resampy scipy numba llvmlite

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
> VBD_ROOT="$(VBD_ROOT)" $(ACTIVATE) && bash scripts/build_vbd_pairs.sh

# verificări rapide VoiceBank + LibriSpeech (manifest + număr fișiere, rate etc.)
data-verify: check-scripts
> $(ACTIVATE) && bash scripts/verify_and_prepare.sh

# ---------- Noise lifecycle ----------
noise.ensure:
> chmod +x scripts/ensure_noise.sh
> $(ACTIVATE) && bash scripts/ensure_noise.sh --demand "$(DEMAND_ROOT)" --min-count 30 --sr 16000

noise.verify:
> $(ACTIVATE) && python scripts/verify_noise.py

# ---------- LISTS (idempotent; fără circularitate) ----------
LISTS_DIR_STAMP := $(LISTS_DIR)/.ok

$(LISTS_DIR_STAMP):
> @mkdir -p "$(LISTS_DIR)"
> @touch "$@"

$(CLEAN_LIST): | $(LISTS_DIR_STAMP)
> if [ -d "$(LIBRISPEECH_ROOT)" ]; then \
>   find "$(LIBRISPEECH_ROOT)" -type f \( -iname '*.flac' -o -iname '*.wav' \) | sort > "$@"; \
> else \
>   echo "[ERR] LIBRISPEECH_ROOT missing: $(LIBRISPEECH_ROOT)"; exit 2; \
> fi

$(NOISE_LIST): | $(LISTS_DIR_STAMP)
> if [ -d "$(DEMAND_ROOT)" ]; then \
>   find "$(DEMAND_ROOT)" -type f -iname '*.wav' | sort > "$@"; \
> else \
>   echo "[ERR] DEMAND_ROOT missing: $(DEMAND_ROOT)"; exit 2; \
> fi

$(RIRS_LIST): | $(LISTS_DIR_STAMP)
> if [ -d "$(RIRS_ROOT)" ]; then \
>   find "$(RIRS_ROOT)" -type f -iname '*.wav' | sort > "$@"; \
> else \
>   : > "$@"; \
> fi

$(LISTS_STAMP): $(CLEAN_LIST) $(NOISE_LIST) $(RIRS_LIST)
> @mkdir -p "$(STAMP_DIR)"
> @touch "$@"

lists: $(LISTS_STAMP)
> @echo "[lists] OK → $(CLEAN_LIST)  $(NOISE_LIST)  $(RIRS_LIST)"

lists.clean:
> rm -f $(CLEAN_LIST) $(NOISE_LIST) $(RIRS_LIST) $(LISTS_STAMP) $(LISTS_DIR_STAMP)

# ---------- Sources ensure (idempotent via stamps) ----------
$(LIBRISPEECH_STAMP):
> @echo "[datasets] check LibriSpeech at $(LIBRISPEECH_ROOT)"
> if [ -d "$(LIBRISPEECH_ROOT)" ] && find "$(LIBRISPEECH_ROOT)" -type f \( -iname '*.flac' -o -iname '*.wav' \) | grep -q .; then \
>   echo "  -> found"; \
> else \
>   test -f scripts/get_librispeech.sh || { echo "[ERR] scripts/get_librispeech.sh missing"; exit 2; }; \
>   bash scripts/get_librispeech.sh "$(LIBRISPEECH_ROOT)"; \
> fi
> touch $@

$(DEMAND_STAMP):
> @echo "[datasets] check DEMAND at $(DEMAND_ROOT)"
> if [ -d "$(DEMAND_ROOT)" ] && find "$(DEMAND_ROOT)" -type f -iname '*.wav' | grep -q .; then \
>   echo "  -> found"; \
> else \
>   test -f scripts/get_demand.sh || { echo "[ERR] scripts/get_demand.sh missing"; exit 2; }; \
>   bash scripts/get_demand.sh "$(DEMAND_ROOT)"; \
> fi
> touch $@

$(RIRS_STAMP):
> @echo "[datasets] check RIRS at $(RIRS_ROOT)"
> if [ -d "$(RIRS_ROOT)" ] && find "$(RIRS_ROOT)" -type f -iname '*.wav' | grep -q .; then \
>   echo "  -> found"; \
> else \
>   test -f scripts/get_rirs.sh || { echo "[WARN] scripts/get_rirs.sh missing; continuing without RIRS."; }; \
> fi
> touch $@

$(VBD_STAMP):
> @echo "[datasets] check VoiceBank-DEMAND at $(VBD_ROOT)"
> if [ -d "$(VBD_ROOT)" ] && find "$(VBD_ROOT)" -type f -iname '*.wav' | grep -q .; then \
>   echo "  -> found"; \
> else \
>   test -f scripts/get_voicebank_demand.sh || { echo "[ERR] scripts/get_voicebank_demand.sh missing"; exit 2; }; \
>   bash scripts/get_voicebank_demand.sh "$(VBD_ROOT)"; \
> fi
> touch $@

# ---------- Mix generation (idempotent via mixgen verify/manifest) ----------
$(MIX_TRAIN_STAMP): $(LIBRISPEECH_STAMP) $(DEMAND_STAMP) $(RIRS_STAMP) $(LISTS_STAMP)
> @echo "[mix] train → $(MIX_TRAIN_DIR)"
> mkdir -p "$(MIX_TRAIN_DIR)"
> $(ACTIVATE) && python "$(MIXGEN)" \
>   --clean-list "$(CLEAN_LIST)" \
>   --noise-list "$(NOISE_LIST)" \
>   --rir-list   "$(RIRS_LIST)" \
>   --out-dir    "$(MIX_TRAIN_DIR)" \
>   --snr        $(SNR) \
>   --sr         $(SR) \
>   --n-mixes    $(N_MIX_TRAIN) \
>   --seed       $(MIX_SEED) \
>   --reverb-prob $(REVERB_PROB) \
>   --codec      $(CODEC) \
>   --clipping   $(CLIPPING) \
>   --segment-min $(SEG_MIN) \
>   --segment-max $(SEG_MAX)
> $(ACTIVATE) && python "$(MIXGEN)" \
  --clean-list "$(CLEAN_LIST)" \
  --noise-list "$(NOISE_LIST)" \
  --rir-list   "$(RIRS_LIST)" \
  --out-dir    "$(MIX_TRAIN_DIR)" \
  --snr        $(SNR) \
  --sr         $(SR) \
  --verify-only
> touch $@

$(MIX_VAL_STAMP): $(LIBRISPEECH_STAMP) $(DEMAND_STAMP) $(LISTS_STAMP)
> @echo "[mix] val → $(MIX_VAL_DIR)"
> mkdir -p "$(MIX_VAL_DIR)"
> $(ACTIVATE) && python "$(MIXGEN)" \
>   --clean-list "$(CLEAN_LIST)" \
>   --noise-list "$(NOISE_LIST)" \
>   --rir-list   "$(RIRS_LIST)" \
>   --out-dir    "$(MIX_VAL_DIR)" \
>   --snr        $(SNR) \
>   --sr         $(SR) \
>   --n-mixes    $(N_MIX_VAL) \
>   --seed       $(MIX_SEED) \
>   --reverb-prob 0.0 \
>   --codec      $(CODEC) \
>   --clipping   $(CLIPPING) \
>   --segment-min $(SEG_MIN) \
>   --segment-max $(SEG_MAX)
> $(ACTIVATE) && python "$(MIXGEN)" \
  --clean-list "$(CLEAN_LIST)" \
  --noise-list "$(NOISE_LIST)" \
  --rir-list   "$(RIRS_LIST)" \
  --out-dir    "$(MIX_VAL_DIR)" \
  --snr        $(SNR) \
  --sr         $(SR) \
  --verify-only
> touch $@

# ---------- Datasets Orchestration ----------
datasets: deps-audio noise.ensure noise.verify $(MIX_TRAIN_STAMP) $(MIX_VAL_STAMP)
> @echo "[datasets] all good."

datasets.verify: deps-audio
> set -e
> $(ACTIVATE) && python "$(MIXGEN)" \
  --clean-list "$(CLEAN_LIST)" \
  --noise-list "$(NOISE_LIST)" \
  --rir-list   "$(RIRS_LIST)" \
  --out-dir    "$(MIX_TRAIN_DIR)" \
  --snr        $(SNR) \
  --sr         $(SR) \
  --verify-only
> $(ACTIVATE) && python "$(MIXGEN)" --out-dir "$(MIX_VAL_DIR)"   --sr $(SR) --verify-only
> @echo "[datasets.verify] OK"

datasets.clean:
> rm -rf "$(PREPARED_DIR)"

# ---------- Unified dataset (AGGREGATOR only) ----------
dataset-unified:
> test -f "$(UNIFIED_SCRIPT)" || { echo "[ERR] $(UNIFIED_SCRIPT) missing. Add it, apoi rulează din nou."; exit 2; }
> $(ACTIVATE) && python "$(UNIFIED_SCRIPT)"

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
> @echo "LISTS             : $(CLEAN_LIST) $(NOISE_LIST) $(RIRS_LIST)"
> @echo "MIX_TRAIN_DIR     : $(MIX_TRAIN_DIR)"
> @echo "MIX_VAL_DIR       : $(MIX_VAL_DIR)"

check-scripts:
> test -f scripts/build_vbd_pairs.sh || { echo "[ERR] scripts/build_vbd_pairs.sh missing"; exit 2; }
> test -f scripts/verify_and_prepare.sh || { echo "[ERR] scripts/verify_and_prepare.sh missing"; exit 2; }
> test -f scripts/ensure_noise.sh    || { echo "[ERR] scripts/ensure_noise.sh missing"; exit 2; }
> test -f scripts/verify_noise.py    || { echo "[ERR] scripts/verify_noise.py missing"; exit 2; }
> true

# ---------- Cleanup ----------
clean:
> scripts/cleanup.sh --yes || true

clean-all:
> scripts/cleanup.sh --aggressive --yes || true

clean-tb:
> scripts/cleanup.sh --only tb --yes || true
