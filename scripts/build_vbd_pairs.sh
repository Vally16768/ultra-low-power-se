#!/usr/bin/env bash
set -euo pipefail
# Construieste perechi pentru VoiceBank-DEMAND (train + test), suportă *_wav / *_16k.
# VBD_ROOT poate fi dat din .env.local; default = data/voicebank-demand-16k.

: "${VBD_ROOT:=data/voicebank-demand-16k}"

OUT_LIST="data/lists/vbd_pairs_test.csv"
mkdir -p "$(dirname "$OUT_LIST")"

detect_dir() {
  local base="$1"; shift
  local d
  for pat in "$@"; do
    d="$(find "$base" -maxdepth 1 -type d -iname "$pat" | head -n1 || true)"
    [[ -n "$d" ]] && { echo "$d"; return 0; }
  done
  return 1
}

TRAIN_CLEAN="$(detect_dir "$VBD_ROOT" "clean_trainset*_16k" "clean_trainset*_wav" "clean_trainset*")" || true
TRAIN_NOISY="$(detect_dir "$VBD_ROOT" "noisy_trainset*_16k" "noisy_trainset*_wav" "noisy_trainset*")" || true
TEST_CLEAN="$(detect_dir "$VBD_ROOT" "clean_testset*_16k"  "clean_testset*_wav"  "clean_testset*")" || true
TEST_NOISY="$(detect_dir "$VBD_ROOT" "noisy_testset*_16k"  "noisy_testset*_wav"  "noisy_testset*")" || true

echo "[VBD] root: $VBD_ROOT"
echo "  train clean: ${TRAIN_CLEAN:-<missing>}"
echo "  train noisy: ${TRAIN_NOISY:-<missing>}"
echo "  test  clean: ${TEST_CLEAN:-<missing>}"
echo "  test  noisy: ${TEST_NOISY:-<missing>}"

[[ -z "${TEST_CLEAN:-}" || -z "${TEST_NOISY:-}" ]] && { echo "[ERROR] lipsesc folderele test"; exit 2; }

# Scrie test.csv + listă compat
python3 - <<'PY'
import os, csv, glob
VBD=os.environ["VBD_ROOT"]
def pairs(n_dir, c_dir):
  nd={os.path.basename(p):p for p in glob.glob(os.path.join(n_dir,"**","*.wav"),recursive=True)}
  cd={os.path.basename(p):p for p in glob.glob(os.path.join(c_dir,"**","*.wav"),recursive=True)}
  common=sorted(set(nd)&set(cd))
  return [(nd[k],cd[k]) for k in common]

test_pairs = pairs(os.environ["TEST_NOISY"], os.environ["TEST_CLEAN"])
with open(os.path.join(VBD,"test.csv"),"w",newline="") as f:
  w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(test_pairs)

# listă compat (noisy,clean,meta) pt. tool-uri vechi
with open("data/lists/vbd_pairs_test.csv","w",newline="") as f:
  w=csv.writer(f); w.writerow(["noisy","clean","meta"]); w.writerows([(*p,"") for p in test_pairs])

# train (dacă există folderele)
tc, tn = os.environ.get("TRAIN_CLEAN"), os.environ.get("TRAIN_NOISY")
if tc and tn:
  train_pairs = pairs(tn, tc)
  with open(os.path.join(VBD,"train.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(train_pairs)
  print(f"[VBD] train pairs: {len(train_pairs)}")
print(f"[VBD] test  pairs: {len(test_pairs)}")
PY

echo "[build_vbd_pairs] wrote: $VBD_ROOT/train.csv (dacă train exista) & $VBD_ROOT/test.csv"
echo "[build_vbd_pairs] wrote: $OUT_LIST (compat)"
