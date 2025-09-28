#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "⚠️  Nu rula tot scriptul cu sudo. Ieși și rulează: ./scripts/test_ci_local.sh"
  exit 1
fi

log() { printf "\n\033[1;36m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

# pick runner: uv if present, else plain python/pip
if command -v uv >/dev/null 2>&1; then
  PYRUN="uv run"
  PIP="uv pip"
  log "Using uv: $(command -v uv)"
else
  PYRUN="python"
  PIP="python -m pip"
  log "uv not found → falling back to python/pip"
fi

maybe_install_system_deps() {
  if command -v apt-get &>/dev/null; then
    log "Install system deps (libsndfile1)..."
    sudo apt-get update -y || true
    sudo apt-get install -y libsndfile1 || true
  fi
}

maybe_install_system_deps

log "Python version"
$PYRUN - <<'PY'
import sys; print("python:", sys.version.split()[0], "exe:", sys.executable)
PY

# 1) Lint + unit
log "Install deps (lint/unit)"
$PIP install --upgrade pytest ruff mypy numpy types-PyYAML

log "Ruff"
$PYRUN -m ruff check .

log "Mypy"
$PYRUN -m mypy --config-file mypy.ini --install-types --non-interactive

log "Pytest (unit)"
$PYRUN -m pytest -q

# 2) Metrics fixtures
log "Install deps (metrics)"
$PIP install --upgrade numpy soundfile pesq pystoi onnxruntime

if [ -f tests/test_metrics_fixtures.py ]; then
  log "Pytest (metrics fixtures)"
  $PYRUN -m pytest -q tests/test_metrics_fixtures.py
else
  log "tests/test_metrics_fixtures.py absent → skip"
fi

# 3) ONNX sanity (optional)
log "Install deps (onnx)"
$PIP install --upgrade onnx onnxruntime numpy

ONNX_FILE="$(git ls-files '*.onnx' | head -n1 || true)"
if [ -n "${ONNX_FILE}" ]; then
  log "Sanity ONNX on ${ONNX_FILE}"
  $PYRUN deploy/sanity_onnx.py --onnx "${ONNX_FILE}"
else
  log "No ONNX found in repo → skip sanity"
fi

# 4) Train smoke (1 epoch)
log "Deps train"
$PIP install --upgrade 'torch==2.3.*' --index-url https://download.pytorch.org/whl/cpu
$PIP install --upgrade numpy pyyaml tensorboard soundfile

log "Train (smoke)"
ULTRA_SMOKE=1 $PYRUN runners/train.py --config configs/exp_mamba_unet.yaml

# === 5) Export ONNX + sanity + parity quick ===
log "Export ONNX (cu --config)"
$PYRUN deploy/export_onnx.py --config configs/exp_mamba_unet.yaml

ONNX_PATH="artifacts/export/mamba_unet_v0.onnx"
[ -f "$ONNX_PATH" ] || ONNX_PATH="artifacts/model.onnx"
[ -f "$ONNX_PATH" ] || { log "Nu găsesc ONNX exportat"; exit 1; }
log "ONNX found: $ONNX_PATH"

$PYRUN deploy/sanity_onnx.py "$ONNX_PATH" --print-json

# Parity quick necesită liste
CLEAN_LIST="data/lists/dev_clean.txt"
NOISE_LIST="data/lists/noise_unseen.txt"
if [ ! -f "$CLEAN_LIST" ] || [ ! -f "$NOISE_LIST" ]; then
  log "Liste lipsă → încerc scripts/build_lists.sh"
  if [ -x scripts/build_lists.sh ]; then
    bash scripts/build_lists.sh
  fi
fi
[ -f "$CLEAN_LIST" ] || { log "Missing $CLEAN_LIST"; exit 1; }
[ -f "$NOISE_LIST" ] || { log "Missing $NOISE_LIST"; exit 1; }

$PYRUN eval/bench_onnx.py \
  --model "$ONNX_PATH" \
  --clean-list "$CLEAN_LIST" \
  --noise-list "$NOISE_LIST" \
  --n 8 --secs 2.0 \
  --out artifacts/bench_quick.json \
  --metrics pesq stoi si_sdr delta_snr_db

# === 9) Quantize + rescore (corect cu --model) ===
# === 9) Quantize + rescore ===
log "Quantize + rescore (PTQ static)"
# Ne asigurăm că există listele:
[ -f data/lists/train_clean.txt ] || bash scripts/build_lists.sh
[ -f data/lists/noise_train.txt ] || bash scripts/build_lists.sh

$PYRUN deploy/quantize.py \
  --model "$ONNX_PATH" \
  --static \
  --calib-list data/lists/train_clean.txt \
  --noisy-list data/lists/noise_train.txt \
  --num-calib 40 \
  --method percentile \
  --percentile 99.9 \
  --per-channel-weights \
  --auto-keep-boundary \
  --out artifacts/model_int8.onnx

$PYRUN runners/infer.py --onnx artifacts/model_int8.onnx --mode offline
$PYRUN runners/score.py --ref data/test/clean --est outputs/offline --out artifacts/metrics_int8.json

# === 10) Report HTML ===
log "Build report.html"
$PYRUN reports/build_report.py \
  --fp32 artifacts/metrics.json \
  --int8 artifacts/metrics_int8.json \
  --out artifacts/report.html

# === 11) Gates (ABS + optional GOLDEN) ===
log "Gate ABS/GOLDEN"
THRESH_PESQ_WB="${THRESH_PESQ_WB:-3.00}"
THRESH_STOI="${THRESH_STOI:-0.93}"
THRESH_DELTA_SNR="${THRESH_DELTA_SNR:-9.0}"
THRESH_SI_SDR="${THRESH_SI_SDR:-12.0}"
TOL_PESQ_WB="${TOL_PESQ_WB:-0.01}"
TOL_STOI="${TOL_STOI:-0.001}"
TOL_DELTA_SNR="${TOL_DELTA_SNR:-0.10}"
TOL_SI_SDR="${TOL_SI_SDR:-0.10}"

python3 - <<PY
import json, os, sys, pathlib
m=json.load(open('artifacts/metrics.json'))
mm=m.get("metrics",m)
thr={"pesq_wb":float(os.environ["THRESH_PESQ_WB"]),
     "stoi":float(os.environ["THRESH_STOI"]),
     "delta_snr_db":float(os.environ["THRESH_DELTA_SNR"]),
     "si_sdr_db":float(os.environ["THRESH_SI_SDR"])}
fails=[]
for k,vmin in thr.items():
    v=float(mm.get(k,-1e9))
    print(f"[ABS] {k}: {v:.4f} >= {vmin:.4f} ?", "OK" if v>=vmin else "FAIL")
    if v<vmin: fails.append((k,v,vmin))
if fails: print("[ABS] FAIL:",fails); sys.exit(1)
print("[ABS] PASS")
if pathlib.Path('.github/golden_metrics.json').exists():
    cur=mm; golden=json.load(open('.github/golden_metrics.json'))
    golden=golden.get("metrics",golden)
    tol={"pesq_wb":float(os.environ["TOL_PESQ_WB"]),
         "stoi":float(os.environ["TOL_STOI"]),
         "delta_snr_db":float(os.environ["TOL_DELTA_SNR"]),
         "si_sdr_db":float(os.environ["TOL_SI_SDR"])}
    fails=[]
    for k,t in tol.items():
        d=float(cur.get(k,-1e9))-float(golden.get(k,-1e9))
        print(f"[GOLDEN] {k}: Δ={d:+.4f} (accept ≥ {-t:+.4f})", "OK" if d>=-t else "FAIL")
        if d<-t: fails.append((k,d))
    if fails: print("[GOLDEN] FAIL:",fails); sys.exit(1)
    print("[GOLDEN] PASS")
else:
    print("[GOLDEN] skipped (no .github/golden_metrics.json)")
PY