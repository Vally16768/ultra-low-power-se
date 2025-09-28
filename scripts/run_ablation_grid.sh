#!/usr/bin/env bash
# scripts/run_ablation_grid.sh
# Grilă ablații: {baseline, +quant_static, +quant_dynamic (safe), +weight_only, [ +prune, +prune+quant ]}
# Robusta la diferențe de versiuni ONNX Runtime (fallback, skip modele neloadabile).

set -euo pipefail

# -------------------- Intrări & defaulturi --------------------
MODEL_IN=${1:-artifacts/export/mamba_unet_v0.onnx}
OUTDIR=${2:-results/ablatii}

CONFIG=${CONFIG:-configs/exp_mamba_unet.yaml}
FACTORY=${FACTORY:-se_models.mamba_unet.model:build_model}

CALIB_CLEAN=${CALIB_CLEAN:-data/lists/dev_clean.txt}
CALIB_NOISE=${CALIB_NOISE:-data/lists/noise_dev.txt}
NUM_CALIB=${NUM_CALIB:-256}

# Metodă calibrare pentru PTQ static: minmax (stabil) sau percentile
METHOD=${METHOD:-minmax}          # schimbă în "percentile" dacă vrei; fallback la minmax e în deploy/quantize.py
PERCENTILE=${PERCENTILE:-99.9}

# Parametri benchmark
SR=${SR:-16000}
SECS=${SECS:-2.0}
N=${N:-100}
LATENCY_RUNS=${LATENCY_RUNS:-150}
THREADS=${THREADS:-8}
WARMUP=${WARMUP:-10}
PEAK_NORM=${PEAK_NORM:-0.95}      # 0.0 = off; 0.95 recomandat contra clipping

# -------------------- Autodetect CKPT --------------------
if [[ -z "${CKPT:-}" ]]; then
  if compgen -G "artifacts/ckpts/*.pt" > /dev/null; then
    CKPT=$(ls -t artifacts/ckpts/*.pt | head -n1)
  else
    CKPT=""
  fi
fi

mkdir -p "$OUTDIR"

# -------------------- 0) Baseline --------------------
echo "[0] baseline → $OUTDIR/baseline.onnx"
cp "$MODEL_IN" "$OUTDIR/baseline.onnx"

# -------------------- 1) PTQ static (robust) --------------------
echo "[1] +quant (PTQ static, robust) → $OUTDIR/quant_int8.onnx"
STATIC_ARGS=(
  --model "$MODEL_IN" --static
  --calib-list "$CALIB_CLEAN" --noisy-list "$CALIB_NOISE"
  --num-calib "$NUM_CALIB"
  --auto-keep-boundary
  --per-channel-weights
  --out "$OUTDIR/quant_int8.onnx"
)
if [[ "$METHOD" == "percentile" ]]; then
  STATIC_ARGS+=( --method percentile --percentile "$PERCENTILE" )
else
  STATIC_ARGS+=( --method "$METHOD" )
fi
python deploy/quantize.py "${STATIC_ARGS[@]}"

# -------------------- 1b) PTQ dynamic (safe pe CPU fără ConvInteger) --------------------
echo "[1b] +quant (PTQ dynamic) → $OUTDIR/quant_int8_dyn.onnx"
python deploy/quantize.py --model "$MODEL_IN" --dynamic \
  --auto-keep-boundary \
  --per-channel-weights \
  --out "$OUTDIR/quant_int8_dyn.onnx"

# -------------------- 1c) Weight-only (w8/a32) --------------------
echo "[1c] +quant (weight-only) → $OUTDIR/quant_w8_a32.onnx"
python deploy/quantize.py --model "$MODEL_IN" --static \
  --weight-only \
  --auto-keep-boundary \
  --per-channel-weights \
  --out "$OUTDIR/quant_w8_a32.onnx"

VARIANTS=("baseline" "quant_int8" "quant_int8_dyn" "quant_w8_a32")

# -------------------- 2) Pruning + fine-tune scurt (opțional, dacă avem CKPT) --------------------
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
  PRUNE_STATIC_ARGS=(
    --model "$OUTDIR/prune30.onnx" --static
    --calib-list "$CALIB_CLEAN" --noisy-list "$CALIB_NOISE"
    --num-calib "$NUM_CALIB"
    --auto-keep-boundary
    --per-channel-weights
    --out "$OUTDIR/prune30_quant.onnx"
  )
  if [[ "$METHOD" == "percentile" ]]; then
    PRUNE_STATIC_ARGS+=( --method percentile --percentile "$PERCENTILE" )
  else
    PRUNE_STATIC_ARGS+=( --method "$METHOD" )
  fi
  python deploy/quantize.py "${PRUNE_STATIC_ARGS[@]}"

  VARIANTS+=("prune30" "prune30_quant")
else
  echo "[warn] CKPT inexistent — sar peste prune. Setează CKPT=/cale/model.pt pentru prune."
fi

# -------------------- 4) Sanity + benchmark --------------------
echo "[4] sanity + benchmark"

can_load() {
python - "$1" << 'PY'
import sys, onnxruntime as ort
m = sys.argv[1]
try:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort.InferenceSession(m, sess_options=so, providers=["CPUExecutionProvider"])
except Exception:
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
    --n "$N" \
    --metrics pesq stoi snr sisnr \
    --latency-runs "$LATENCY_RUNS" \
    --threads "$THREADS" \
    --sr "$SR" \
    --secs "$SECS" \
    --warmup "$WARMUP" \
    --peak-norm "$PEAK_NORM" \
    --out "$OUTDIR/${m}_metrics.json"
done

# -------------------- 5) Agregare --------------------
echo "[5] agregare → $OUTDIR/summary.csv"
python eval/aggregate_grid.py --dir "$OUTDIR" --out "$OUTDIR/summary.csv"

echo "[OK] Rezultate în $OUTDIR"
