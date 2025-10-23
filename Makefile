# ------- Config -------
TF_MODEL ?= artifacts/tf_manifest_only/model.keras
NOISY_DIR ?= data/noisy
CLEAN_DIR ?= data/clean
ONNX_DIR  ?= artifacts/tf_manifest_only/onnx
ONNX      ?= $(ONNX_DIR)/unet1d_fp32.onnx
SR        ?= 16000
FRAME_S   ?= 2.0
HOP_S     ?= 1.0
LAYOUT    ?= channels_last
OPSET     ?= 17
OUT_DIR   ?= artifacts/onnx_out

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION ?= cpp

# ------- Phonies -------
.PHONY: all export sanity parity enhance eval demo clean

all: export sanity parity enhance eval

export:
	python scripts/export_to_onnx.py \
	  --src "$(TF_MODEL)" \
	  --out "$(ONNX)" \
	  --opset $(OPSET) \
	  --layout $(LAYOUT) \
	  --verbose

sanity:
	python scripts/sanity_onnx.py --model "$(ONNX)"

parity:
	python scripts/parity_check.py \
	  --tf_model "$(TF_MODEL)" \
	  --onnx "$(ONNX)" \
	  --seconds 2.0 \
	  --sr $(SR)

enhance:
	python scripts/run_onnx_enhance.py \
	  --onnx "$(ONNX)" \
	  --in_path "$(NOISY_DIR)" \
	  --out_dir "$(OUT_DIR)" \
	  --sr $(SR) \
	  --frame_s $(FRAME_S) \
	  --hop_s $(HOP_S) \
	  --batch 8

eval:
	python scripts/eval_quality.py \
	  --clean_dir "$(CLEAN_DIR)" \
	  --noisy_dir "$(NOISY_DIR)" \
	  --enh_dir   "$(OUT_DIR)" \
	  --sr $(SR) \
	  --out_json artifacts/onnx_eval/metrics.json

demo:
	./deploy/gen_dummy_audio.sh

clean:
	rm -rf artifacts/onnx_out artifacts/onnx_eval artifacts/onnx_parity
