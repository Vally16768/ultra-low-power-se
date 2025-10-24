#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
PROJ_DIR="$HOME/projects/ultra-low-power-se"
DATA_DIR="${PROJ_DIR}/data"
mkdir -p "${DATA_DIR}"

# ================= VOICEBANK-DEMAND =================
echo "[INFO] Downloading VoiceBank-DEMAND (16kHz) from Kaggle..."

# Kaggle CLI
if ! command -v kaggle >/dev/null 2>&1; then
  pip install --user kaggle >/dev/null
  export PATH="$PATH:$HOME/.local/bin"
fi

# Kaggle token
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "[ERROR] Lipseste $HOME/.kaggle/kaggle.json (creează din Kaggle -> Settings -> Create New API Token)"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

VBD_DIR="${DATA_DIR}/voicebank-demand-16k"
mkdir -p "${VBD_DIR}"

(
  kaggle datasets download -d jweiqi/voicebank-demand-16k -p "${VBD_DIR}" -q
  ZIP_PATH="$(ls -1 "${VBD_DIR}"/voicebank-demand-16k.zip 2>/dev/null || true)"
  if [ -z "${ZIP_PATH}" ]; then
    # fallback: dacă Kaggle schimbă numele
    ZIP_PATH="$(ls -1 "${VBD_DIR}"/*.zip 2>/dev/null | head -n1 || true)"
  fi
  if [ -z "${ZIP_PATH}" ]; then
    echo "[ERROR] Nu am găsit arhiva Zip descărcată de la Kaggle în ${VBD_DIR}."
    echo "        Verifică dacă ai acceptat terms pe pagina datasetului."
    exit 1
  fi
  unzip -o "${ZIP_PATH}" -d "${VBD_DIR}" >/dev/null
  rm -f "${ZIP_PATH}"
  echo "[OK] VoiceBank-DEMAND extracted to ${VBD_DIR}"
)&

# ================= Wait for all parallel jobs =================
wait
echo "[DONE] All datasets downloaded and extracted."

# ================= Build VoiceBank manifests =================
TRAIN_CLEAN="$(find "${VBD_DIR}" -maxdepth 1 -type d -iname "clean_trainset*_16k" | head -n1)"
TRAIN_NOISY="$(find "${VBD_DIR}" -maxdepth 1 -type d -iname "noisy_trainset*_16k" | head -n1)"
TEST_CLEAN="$(find "${VBD_DIR}" -maxdepth 1 -type d -iname "clean_testset*_16k" | head -n1)"
TEST_NOISY="$(find "${VBD_DIR}" -maxdepth 1 -type d -iname "noisy_testset*_16k" | head -n1)"

if [[ -z "${TRAIN_CLEAN}" || -z "${TRAIN_NOISY}" || -z "${TEST_CLEAN}" || -z "${TEST_NOISY}" ]]; then
  echo "[ERROR] Nu am găsit folderele VoiceBank așteptate în ${VBD_DIR}."
  find "${VBD_DIR}" -maxdepth 2 -type d -print | sed 's/^/[DBG] /'
  exit 1
fi

