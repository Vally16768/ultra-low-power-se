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

# ================= LIBRISPEECH ======================
echo "[INFO] Downloading LibriSpeech subsets in parallel..."

LIBRI_DIR="${DATA_DIR}/librispeech"
mkdir -p "${LIBRI_DIR}"
cd "${LIBRI_DIR}"

BASE_URL="https://www.openslr.org/resources/12"
FILES=(
  "train-clean-100.tar.gz"
  "train-clean-360.tar.gz"
  "dev-clean.tar.gz"
  "test-clean.tar.gz"
)

for f in "${FILES[@]}"; do
(
  if [ ! -f "$f" ]; then
    echo "[DL] $f"
    wget -q -c "${BASE_URL}/${f}" -O "$f"
  else
    echo "[SKIP] $f (already exists)"
  fi
  echo "[UNTAR] $f"
  tar -xzf "$f"
  rm -f "$f"
  echo "[OK] extracted $f"
)&
done

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

export VBD_DIR TRAIN_CLEAN TRAIN_NOISY TEST_CLEAN TEST_NOISY
python3 - <<'PY'
import os, csv, glob, sys

def build_pairs(noisy_dir, clean_dir):
    nd = {os.path.basename(p): p for p in glob.glob(os.path.join(noisy_dir, "**", "*.wav"), recursive=True)}
    cd = {os.path.basename(p): p for p in glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True)}
    common = sorted(set(nd) & set(cd))
    if not common:
        print(f"[ERROR] Nu am găsit perechi comune între\n  noisy={noisy_dir}\n  clean={clean_dir}", file=sys.stderr)
    return [(nd[k], cd[k]) for k in common]

base = os.environ["VBD_DIR"]
pairs_tr = build_pairs(os.environ["TRAIN_NOISY"], os.environ["TRAIN_CLEAN"])
pairs_te = build_pairs(os.environ["TEST_NOISY"], os.environ["TEST_CLEAN"])

for name, pairs in [("train.csv", pairs_tr), ("test.csv", pairs_te)]:
    path = os.path.join(base, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noisy","clean"])
        w.writerows(pairs)
    print(f"[OK] wrote {path} with {len(pairs)} pairs")
PY

# ================= LibriSpeech file list =================
find "${LIBRI_DIR}/LibriSpeech" -type f -name "*.flac" > "${LIBRI_DIR}/clean_files.lst" || true
wc -l "${LIBRI_DIR}/clean_files.lst" 2>/dev/null | awk '{print "[OK] LibriSpeech clean files:", $1}'
echo "[ALL GOOD] Data ready under: ${DATA_DIR}"
