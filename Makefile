.PHONY: setup train eval enhance export score onnx-sanity

VENV ?= .venv

setup:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate; pip install -U pip wheel setuptools
	. $(VENV)/bin/activate; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	. $(VENV)/bin/activate; pip install pyyaml tensorboard tqdm soundfile numpy pesq pystoi onnx onnxruntime

train:
	. $(VENV)/bin/activate; python -m se_cli.cli train --config configs/exp_robustnet_plus.yaml

eval:
	. $(VENV)/bin/activate; python -m se_cli.cli eval --config configs/exp_robustnet_plus.yaml

enhance:
	. $(VENV)/bin/activate; python -m se_cli.cli enhance --config configs/exp_robustnet_plus.yaml -o inference.in_wav=$(IN_WAV) -o inference.out_wav=$(OUT_WAV)

export:
	. $(VENV)/bin/activate; python -m se_cli.cli export --config configs/exp_robustnet_plus.yaml

score:
	. $(VENV)/bin/activate; python -m se_cli.cli score --config configs/exp_robustnet_plus.yaml

onnx-sanity:
	. $(VENV)/bin/activate; python - <<'PY'\
import onnx, onnxruntime as ort, numpy as np\
m='artifacts/export/robustnet_plus.onnx'\
onnx.checker.check_model(m)\
sess=ort.InferenceSession(m, providers=['CPUExecutionProvider'])\
x=np.random.randn(1,1,16000).astype(np.float32)\
y=sess.run(None, {'input': x})[0]\
print('[onnx-sanity]', y.shape)\
PY
