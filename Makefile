.PHONY: help setup test lint fmt data-train train export infer eval onnx-sanity

.PHONY: clean clean-all clean-tb
clean:
	@scripts/cleanup.sh --yes

clean-all:
	@scripts/cleanup.sh --aggressive --yes

clean-tb:
	@scripts/cleanup.sh --only tb --yes

help:
	@echo "Targets:"
	@echo "  setup        - create venv & install deps"
	@echo "  test         - run unit tests"
	@echo "  lint         - ruff + mypy (non-strict)"
	@echo "  fmt          - ruff --fix"
	@echo "  data-train   - prepare VoiceBank+DEMAND lists"
	@echo "  train        - run training via se"
	@echo "  export       - export ONNX"
	@echo "  onnx-sanity  - parity Torch vs ONNX"
	@echo "  clean        - remove artifacts (not models)"
	@echo "  clean-all    - remove all artifacts (including models)"
	@echo "  clean-tb     - remove TensorBoard logs"

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check . && mypy --ignore-missing-imports se_cli runners deploy datasets se_models

fmt:
	. .venv/bin/activate && ruff check --fix .

data-train:
	bash scripts/build_vbd_pairs.sh

train:
	. .venv/bin/activate && python -m se_cli.cli train --config configs/exp_mamba_unet.yaml

export:
	. .venv/bin/activate && python -m se_cli.cli export --config configs/exp_mamba_unet.yaml

onnx-sanity:
	. .venv/bin/activate && python deploy/sanity_onnx.py artifacts/export/mamba_unet_v0.onnx