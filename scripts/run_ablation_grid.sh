#!/usr/bin/env bash
set -euo pipefail

MODEL_IN=${1:-artifacts/export/mamba_unet_v0.onnx}
OUTDIR=${2:-results/ablatii}

CONFIG=${CONFIG:-configs/exp_mamba_unet.yaml}
FACTORY=${FACTORY:-se_models.mamba_unet.model:build_model}

CALIB_CLEAN=${CALIB_CLEAN:-data/lists/dev_clean.txt}
CALIB_NOISE=${CALIB_NOISE:-data/lists/noise_dev.txt}
NUM_CALIB=${NUM_CALIB:-256}  # ↑ calib mai mare ajută

# autodetect CKPT dacă nu e dat
if [[ -z "${CKPT:-}" ]]; then
  if compgen -G "artifacts/ckpts/*.pt" > /dev/null; then
    CKPT=$(ls -t artifacts/ckpts/*.pt | head -n1)
  else
    CKPT=""
  fi
fi

mkdir -p "$OUTDIR"

echo "[0] baseline → $OUTDIR/baseline.onnx"
cp "$MODEL_IN" "$OUTDIR/baseline.onnx"

echo "[1] +quant (PTQ static, robust) → $OUTDIR/quant_int8.onnx"
python deploy/quantize.py --model "$MODEL_IN" --static \
  --calib-list "$CALIB_CLEAN" --noisy-list "$CALIB_NOISE" \
  --num-calib "$NUM_CALIB" \
  --auto-keep-boundary \
  --method percentile --percentile 99.9 \
  --per-channel-weights \
  --out "$OUTDIR/quant_int8.onnx"

echo "[1b] +quant (PTQ dynamic) → $OUTDIR/quant_int8_dyn.onnx"
python deploy/quantize.py --model "$MODEL_IN" --dynamic \
  --auto-keep-boundary \
  --per-channel-weights \
  --out "$OUTDIR/quant_int8_dyn.onnx"

VARIANTS=("baseline" "quant_int8" "quant_int8_dyn")

if [[ -n "$CKPT" && -f "$CKPT" ]]; then
  echo "[2] +prune (30%) + FT scurt → $OUTDIR/prune30.onnx"
  python deploy/prune.py \
    --train-ckpt "$CKPT" \
    --config "$CONFIG" \
    --factory "$FACTORY" \
    --pct 0.3 --epochs 2 \
    --out-ckpt artifacts/ckpts/mamba_unet_pruned.pt \
    --export-onnx "$OUTDIR/prune30.onnx"

  echo "[3] +prune + quant (static robust) → $OUTDIR/prune30_quant.onnx"
  python deploy/quantize.py --model "$OUTDIR/prune30.onnx" --static \
    --calib-list "$CALIB_CLEAN" --noisy-list "$CALIB_NOISE" \
    --num-calib "$NUM_CALIB" \
    --auto-keep-boundary \
    --method percentile --percentile 99.9 \
    --per-channel-weights \
    --out "$OUTDIR/prune30_quant.onnx"

  VARIANTS+=("prune30" "prune30_quant")
else
  echo "[warn] CKPT inexistent — sar peste prune. Setează CKPT=/cale/model.pt pentru prune."
fi

echo "[4] sanity + benchmark"

can_load() {
python - "$1" << 'PY'
import sys, onnxruntime as ort
m = sys.argv[1]
try:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort.InferenceSession(m, sess_options=so, providers=["CPUExecutionProvider"])
except Exception as e:
    sys.exit(2)
print("OK")
PY
}

for m in "${VARIANTS[@]}"; do
  MODEL_PATH="$OUTDIR/${m}.onnx"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[skip] missing $MODEL_PATH"
    continue
  fi

  if ! can_load "$MODEL_PATH" >/dev/null 2>&1; then
    echo "[skip] $m: ORT cannot load this model on CPU EP (likely ConvInteger)."
    continue
  fi

  python deploy/sanity_onnx.py "$MODEL_PATH" --print-json > "$OUTDIR/${m}_sanity.json" || true
  python eval/bench_onnx.py \
    --model "$MODEL_PATH" \
    --clean-list "$CALIB_CLEAN" \
    --noise-list "$CALIB_NOISE" \
    --n 100 \
    --metrics pesq stoi snr sisnr \
    --latency-runs 150 \
    --threads 8 \
    --sr 16000 \
    --warmup 10 \
    --out "$OUTDIR/${m}_metrics.json"
done

echo "[5] agregare → $OUTDIR/summary.csv"
python eval/aggregate_grid.py --dir "$OUTDIR" --out "$OUTDIR/summary.csv"

echo "[OK] Rezultate în $OUTDIR"
