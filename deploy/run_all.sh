#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
TF_MODEL="${1:-artifacts/tf_manifest_only/model.keras}"   # .keras/.h5 or SavedModel dir
NOISY_DIR="${2:-data/noisy}"       # folder of noisy audio (recursive ok)
CLEAN_DIR="${3:-data/clean}"       # parallel clean refs (same stems)
OUT_DIR="artifacts/onnx_out"
ONNX_DIR="artifacts/tf_manifest_only/onnx"
ONNX="${ONNX_DIR}/unet1d_fp32.onnx"
SR=16000
FIXED_LEN=""          # e.g. 32000 for fixed T; leave empty for dynamic
LAYOUT="channels_last"  # channels_last -> [B,T,1], channels_first -> [B,1,T]
OPSET=17
FRAME_S=2.0
HOP_S=1.0

# For faster protobuf conversions if available:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-cpp}

echo "== 1) Export to ONNX =="
python scripts/export_to_onnx.py \
  --src "${TF_MODEL}" \
  --out "${ONNX}" \
  --opset "${OPSET}" \
  ${FIXED_LEN:+--fixed_len "$FIXED_LEN"} \
  --layout "${LAYOUT}" \
  --verbose

echo ""
echo "== 2) Sanity check ONNX =="
python scripts/sanity_onnx.py --model "${ONNX}"

echo ""
echo "== 3) TF vs ONNX parity check (random input @ ${SR} Hz) =="
python scripts/parity_check.py \
  --tf_model "${TF_MODEL}" \
  --onnx "${ONNX}" \
  --seconds 2.0 \
  --sr "${SR}"

# ---- Enhance only if we have input audio ----
echo ""
echo "== 4) Enhance audio with ONNX =="
if [ -d "${NOISY_DIR}" ] && compgen -G "${NOISY_DIR}/**/*.*" > /dev/null; then
  python scripts/run_onnx_enhance.py \
    --onnx "${ONNX}" \
    --in_path "${NOISY_DIR}" \
    --out_dir "${OUT_DIR}" \
    --sr "${SR}" \
    --frame_s "${FRAME_S}" \
    --hop_s "${HOP_S}" \
    --batch 8
else
  echo "[SKIP] No audio found under '${NOISY_DIR}'. Skipping enhancement."
fi

# ---- Evaluate only if clean+enhanced exist ----
echo ""
echo "== 5) Objective eval (SI-SNRi, SNRi, optional PESQ/STOI) =="
if [ -d "${CLEAN_DIR}" ] && [ -d "${OUT_DIR}" ] && compgen -G "${CLEAN_DIR}/**/*.*" > /dev/null && compgen -G "${OUT_DIR}/**/*.wav" > /dev/null; then
  python scripts/eval_quality.py \
    --clean_dir "${CLEAN_DIR}" \
    --noisy_dir "${NOISY_DIR}" \
    --enh_dir   "${OUT_DIR}" \
    --sr "${SR}" \
    --out_json "artifacts/onnx_eval/metrics.json"
else
  echo "[SKIP] Missing clean/enhanced audio. Skipping evaluation."
fi

echo ""
echo "All done. Outputs:"
echo " - ONNX: ${ONNX}"
echo " - Enhanced WAVs (if any): ${OUT_DIR}"
echo " - Metrics JSON (if eval ran): artifacts/onnx_eval/metrics.json"
