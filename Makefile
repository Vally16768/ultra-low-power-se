# -----------------------------
# Ultra-Low-Power SE — Makefile
# -----------------------------
.RECIPEPREFIX := >
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

# Încarcă variabile locale dacă există
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

# Dataset roots (pot fi suprascrise din .env.local)
LIBRISPEECH_ROOT ?= $(PWD)/data/librispeech/LibriSpeech
DEMAND_ROOT      ?= $(PWD)/data/noise/demand
RIRS_ROOT        ?= $(PWD)/data/rirs
VBD_ROOT         ?= $(PWD)/data/voicebank-demand-16k
MIXGEN           ?= datasets/mixgen.py

# Parametri generare
SR            ?= 16000
N_TRAIN       ?= 2000
N_DEV         ?= 300
N_CHALLENGE   ?= 500
SNR_TRAIN_DEV ?= -5 0 5 10 15   # IMPORTANT: spații, nu virgule
SNR_CHALLENGE ?= -5 0 5 10

# Layout de ieșire
OUT_TRAIN     ?= data/prepared/train_mixes
OUT_DEV       ?= data/prepared/dev_mixes
OUT_TEST      ?= data/prepared/voicebank/test
OUT_CHALLENGE ?= data/prepared/test_challenge

# ---------- PHONY ----------
.PHONY: setup train eval enhance export score onnx-sanity \
        datasets datasets.verify datasets.clean data-verify data-train dataset-unified

# ---------- VENV & toolchain ----------
setup:
>	$(PYTHON) -m venv $(VENV)
>	$(ACTIVATE); pip install -U pip wheel setuptools
>	$(ACTIVATE); pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
>	$(ACTIVATE); pip install pyyaml tensorboard tqdm soundfile numpy pesq pystoi onnx onnxruntime resampy

# ---------- Rulări model ----------
train:
>	$(ACTIVATE); python -m se_cli.cli train --config configs/exp_robustnet_plus.yaml

eval:
>	$(ACTIVATE); python -m se_cli.cli eval --config configs/exp_robustnet_plus.yaml

enhance:
>	$(ACTIVATE); python -m se_cli.cli enhance --config configs/exp_robustnet_plus.yaml -o inference.in_wav=$(IN_WAV) -o inference.out_wav=$(OUT_WAV)

export:
>	$(ACTIVATE); python -m se_cli.cli export --config configs/exp_robustnet_plus.yaml

score:
>	$(ACTIVATE); python -m se_cli.cli score --config configs/exp_robustnet_plus.yaml

onnx-sanity:
>	$(ACTIVATE); python - <<'PY'
> import onnx, onnxruntime as ort, numpy as np
> m='artifacts/export/robustnet_plus.onnx'
> onnx.checker.check_model(m)
> sess=ort.InferenceSession(m, providers=['CPUExecutionProvider'])
> x=np.random.randn(1,1,16000).astype(np.float32)
> y=sess.run(None, {'input': x})[0]
> print('[onnx-sanity]', y.shape)
> PY

# ============================================================
#                    DATASETS PIPELINE
# ============================================================

# Pipeline complet: liste + VoiceBank test + mixgen(train/dev/challenge) + manifeste + verificări
datasets:
>	@echo "==[lists]=="
>	mkdir -p data/lists $(OUT_TRAIN) $(OUT_DEV) $(OUT_TEST)/manifests $(OUT_CHALLENGE)
>	# Clean (LibriSpeech): train + dev (subset)
>	find "$(LIBRISPEECH_ROOT)" -type f \( -iname '*.flac' -o -iname '*.wav' \) | sort > data/lists/train_clean.txt
>	head -n 1000 data/lists/train_clean.txt > data/lists/dev_clean.txt || true
>	# Noise: DEMAND + alte surse dacă există
>	: > data/lists/noise_train.txt
>	if [ -d "$(DEMAND_ROOT)" ]; then find "$(DEMAND_ROOT)" -type f -iname '*.wav' >> data/lists/noise_train.txt; fi
>	for d in data/noise/custom data/noise/dev data/noise/synth; do \
>	  if [ -d "$$d" ]; then find "$$d" -type f -iname '*.wav' >> data/lists/noise_train.txt; fi; \
>	done
>	sort -o data/lists/noise_train.txt data/lists/noise_train.txt
>	head -n 300 data/lists/noise_train.txt > data/lists/noise_dev.txt || true
>	# Unseen noise (opțional)
>	if [ -d data/noise/unseen ]; then find data/noise/unseen -type f -iname '*.wav' | sort > data/lists/noise_unseen.txt; fi
>	# RIRs (opțional)
>	if [ -d "$(RIRS_ROOT)" ]; then find "$(RIRS_ROOT)" -type f -iname '*.wav' | sort > data/lists/rir_list.txt; fi
>
>	@echo "==[vbd test] symlink + manifest=="
>	[ -e "$(OUT_TEST)/noisy" ] || ln -s "$$(realpath "$(VBD_ROOT)/noisy_testset_wav")" "$(OUT_TEST)/noisy"
>	[ -e "$(OUT_TEST)/clean" ] || ln -s "$$(realpath "$(VBD_ROOT)/clean_testset_wav")" "$(OUT_TEST)/clean"
>	$(ACTIVATE); python - <<'PY'
> import csv
> from pathlib import Path
> out = Path("data/prepared/voicebank/test"); out.mkdir(parents=True, exist_ok=True)
> pairs = out/"manifests/pairs.csv"; pairs.parent.mkdir(parents=True, exist_ok=True)
> noisy, clean = out/"noisy", out/"clean"
> idx = {p.name:p for p in clean.glob("*.wav")}
> with pairs.open("w", newline="") as f:
>     w=csv.writer(f); w.writerow(["noisy","clean"])
>     for n in sorted(noisy.glob("*.wav")):
>         c=idx.get(n.name)
>         if c: w.writerow([str(n.resolve()), str(c.resolve())])
> print("[test] wrote manifest:", pairs)
> PY
>
>	@echo "==[mixgen] train/dev=="
>	$(ACTIVATE); python "$(MIXGEN)" \
>	  --clean-list data/lists/train_clean.txt \
>	  --noise-list data/lists/noise_train.txt \
>	  --out-dir   "$(OUT_TRAIN)" \
>	  --n-mixes   $(N_TRAIN) \
>	  --snr       $(SNR_TRAIN_DEV) \
>	  --sr        $(SR)
>	$(ACTIVATE); python "$(MIXGEN)" \
>	  --clean-list data/lists/dev_clean.txt \
>	  --noise-list data/lists/noise_dev.txt \
>	  --out-dir   "$(OUT_DEV)" \
>	  --n-mixes   $(N_DEV) \
>	  --snr       $(SNR_TRAIN_DEV) \
>	  --sr        $(SR)
>
>	@echo "==[mixgen] challenge (unseen noise dacă există)=="
>	if [ -f data/lists/noise_unseen.txt ]; then \
>	  NL="data/lists/noise_unseen.txt"; \
>	else \
>	  NL="data/lists/noise_dev.txt"; \
>	fi; \
>	$(ACTIVATE); python "$(MIXGEN)" \
>	  --clean-list data/lists/dev_clean.txt \
>	  --noise-list $$NL \
>	  --out-dir   "$(OUT_CHALLENGE)" \
>	  --n-mixes   $(N_CHALLENGE) \
>	  --snr       $(SNR_CHALLENGE) \
>	  --sr        $(SR)
>
>	@echo "==[manifests] train/dev/challenge=="
>	$(ACTIVATE); python - <<'PY'
> import csv
> from pathlib import Path
> def write_pairs(root:str):
>     r=Path(root); (r/"manifests").mkdir(parents=True, exist_ok=True)
>     noisy=r/"noisy"; clean=r/"clean"; out=r/"manifests/pairs.csv"
>     idx={p.name:p for p in clean.glob("*.wav")}
>     n=0
>     with out.open("w", newline="") as f:
>         w=csv.writer(f); w.writerow(["noisy","clean"])
>         for npath in sorted(noisy.glob("*.wav")):
>             c=idx.get(npath.name)
>             if c: w.writerow([str(npath.resolve()), str(c.resolve())]); n+=1
>     print(f"[pairs] {root}: {n} rows")
> for d in ("$(OUT_TRAIN)","$(OUT_DEV)","$(OUT_CHALLENGE)"): write_pairs(d)
> PY
>
>	@$(MAKE) datasets.verify

# Verificări rapide (fără regenerare): mixgen --verify-only + consistențe manifest
datasets.verify:
>	@echo "==[verify] mixgen manifests/existence=="
>	$(ACTIVATE); python "$(MIXGEN)" --clean-list data/lists/train_clean.txt --noise-list data/lists/noise_train.txt --out-dir "$(OUT_TRAIN)" --snr $(SNR_TRAIN_DEV) --verify-only || true
>	$(ACTIVATE); python "$(MIXGEN)" --clean-list data/lists/dev_clean.txt   --noise-list data/lists/noise_dev.txt   --out-dir "$(OUT_DEV)"   --snr $(SNR_TRAIN_DEV) --verify-only || true
>	@echo "==[verify] counts=="
>	$(ACTIVATE); python - <<'PY'
> from pathlib import Path
> def chk(root):
>   r=Path(root)
>   n_noisy=len(list((r/"noisy").glob("*.wav")))
>   n_clean=len(list((r/"clean").glob("*.wav")))
>   rows=sum(1 for _ in open(r/"manifests/pairs.csv"))-1 if (r/"manifests/pairs.csv").exists() else -1
>   print(f"[check] {root:35s} noisy={n_noisy} clean={n_clean} rows={rows}")
>   assert rows==-1 or rows==min(n_noisy,n_clean)
> for d in ("$(OUT_TRAIN)","$(OUT_DEV)","$(OUT_CHALLENGE)","$(OUT_TEST)"): chk(d)
> PY
>	@echo "[ok] datasets.verify"

# Curăță doar seturile sintetice generate (nu atinge VoiceBank original)
datasets.clean:
>	rm -rf "$(OUT_TRAIN)" "$(OUT_DEV)" "$(OUT_CHALLENGE)"
>	@echo "[clean] removed synthetic prepared sets"

# Reconstruiește manifestele VoiceBank (train/test) în folderul oficial
data-train:
>	@echo "==[VoiceBank train/test manifests]=="
>	$(ACTIVATE); python - <<'PY'
> import csv, os
> from pathlib import Path
> root=Path("$(VBD_ROOT)")
> out_train=root/"train.csv"; out_test=root/"test.csv"
> def write(split_clean, split_noisy, out_csv):
>   cdir=root/split_clean; ndir=root/split_noisy
>   idx={p.name:p for p in (root/split_clean).glob("*.wav")}
>   with (root/out_csv).open("w", newline="") as f:
>     w=csv.writer(f); w.writerow(["noisy","clean"])
>     for n in sorted((root/split_noisy).glob("*.wav")):
>       c=idx.get(n.name)
>       if c: w.writerow([str(n.resolve()), str(c.resolve())])
> write("clean_trainset_28spk_wav","noisy_trainset_28spk_wav","train.csv")
> write("clean_testset_wav","noisy_testset_wav","test.csv")
> print("[vbd] wrote:", out_train, "and", out_test)
> PY

# Verifică VoiceBank + Libri (manifest & counts) sau rulează scriptul de verify dacă există
data-verify:
>	if [ -x datasets/verify_and_prepare.sh ]; then \
>	  bash datasets/verify_and_prepare.sh; \
>	else \
>	  $(ACTIVATE); python - <<'PY'; \
>	from pathlib import Path; import sys; \
>	def must(p): \
>	  p=Path(p); print("[ok]" if p.exists() else "[missing]", p); \
>	  return p.exists() \
>	all_ok=True; \
>	for p in ["$(LIBRISPEECH_ROOT)","$(VBD_ROOT)/clean_testset_wav","$(VBD_ROOT)/noisy_testset_wav"]: \
>	  all_ok = must(p) and all_ok; \
>	sys.exit(0 if all_ok else 1) \
>	PY; \
>	fi

# Builder „unificat” (VBD ∪ augment Libri); dacă scriptul nu există, doar loghează
dataset-unified:
>	if [ -f scripts/build_unified_dataset.py ]; then \
>	  $(ACTIVATE); python scripts/build_unified_dataset.py --mix_per_clean 2 --snr 0,5,10,15; \
>	else \
>	  echo "[skip] scripts/build_unified_dataset.py lipsește — nimic de făcut"; \
>	fi
