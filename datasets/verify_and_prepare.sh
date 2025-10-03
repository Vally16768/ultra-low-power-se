#!/usr/bin/env bash
set -euo pipefail

# === Config de bază ===
PROJ_DIR="$HOME/projects/ultra-low-power-se"
DATA_DIR="${PROJ_DIR}/data"
VBD_DIR="${DATA_DIR}/voicebank-demand-16k"
LIBRI_DIR="${DATA_DIR}/librispeech"

echo "[INFO] Using project: ${PROJ_DIR}"
echo "[INFO] Data root    : ${DATA_DIR}"

if [ ! -d "${VBD_DIR}" ]; then
  echo "[ERROR] Missing ${VBD_DIR}. Rulează întâi downloader-ul."
  exit 1
fi

# === Detectează folderele VoiceBank-DEMAND (suportă *_16k și *_wav) ===
detect_dir() {
  local base="$1"; shift
  local pat
  for pat in "$@"; do
    local d
    d="$(find "$base" -maxdepth 1 -type d -iname "$pat" | head -n1 || true)"
    if [ -n "$d" ]; then echo "$d"; return 0; fi
  done
  return 1
}

TRAIN_CLEAN="$(detect_dir "${VBD_DIR}" "clean_trainset*_16k" "clean_trainset*_wav" "clean_trainset*")" || true
TRAIN_NOISY="$(detect_dir "${VBD_DIR}" "noisy_trainset*_16k" "noisy_trainset*_wav" "noisy_trainset*")" || true
TEST_CLEAN="$(detect_dir "${VBD_DIR}" "clean_testset*_16k"  "clean_testset*_wav"  "clean_testset*")" || true
TEST_NOISY="$(detect_dir "${VBD_DIR}" "noisy_testset*_16k"  "noisy_testset*_wav"  "noisy_testset*")" || true

echo "[INFO] VoiceBank-DEMAND layout:"
echo "  train clean: ${TRAIN_CLEAN:-<missing>}"
echo "  train noisy: ${TRAIN_NOISY:-<missing>}"
echo "  test  clean: ${TEST_CLEAN:-<missing>}"
echo "  test  noisy: ${TEST_NOISY:-<missing>}"

if [[ -z "${TRAIN_CLEAN:-}" || -z "${TRAIN_NOISY:-}" || -z "${TEST_CLEAN:-}" || -z "${TEST_NOISY:-}" ]]; then
  echo "[ERROR] Nu găsesc toate folderele necesare în ${VBD_DIR}."
  find "${VBD_DIR}" -maxdepth 2 -type d -print | sed 's/^/[DBG] /'
  exit 1
fi

# === Rulează verificarea + pregătirea (Python standard library only) ===
export VBD_DIR TRAIN_CLEAN TRAIN_NOISY TEST_CLEAN TEST_NOISY LIBRI_DIR
python3 - <<'PY'
import os, sys, csv, glob, wave, contextlib

def scan_wavs(root):
    files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    info = []
    errors = []
    for p in sorted(files):
        try:
            with contextlib.closing(wave.open(p, 'rb')) as wf:
                sr   = wf.getframerate()
                ch   = wf.getnchannels()
                nfrm = wf.getnframes()
                dur  = nfrm / float(sr) if sr else 0.0
        except Exception as e:
            errors.append((p, f"corrupt/unsupported wav: {e}"))
            sr=ch=0; dur=0.0
        info.append((p, sr, ch, dur))
    return info, errors

def pairs_report(noisy_dir, clean_dir):
    nmap = {os.path.basename(p): p for p in glob.glob(os.path.join(noisy_dir, "**", "*.wav"), recursive=True)}
    cmap = {os.path.basename(p): p for p in glob.glob(os.path.join(clean_dir,  "**", "*.wav"), recursive=True)}
    common = sorted(set(nmap) & set(cmap))
    miss_n = sorted(set(cmap) - set(nmap))
    miss_c = sorted(set(nmap) - set(cmap))
    return [(nmap[k], cmap[k]) for k in common], miss_n, miss_c

def header(msg): print("\n" + msg + "\n" + "-"*len(msg))

def check_split(name, noisy_dir, clean_dir, out_csv, want_sr=16000, want_ch=1):
    header(f"[CHECK] {name}: noisy={noisy_dir} | clean={clean_dir}")
    # Scan WAVs
    ninfo, nerr = scan_wavs(noisy_dir)
    cinfo, cerr = scan_wavs(clean_dir)
    if nerr or cerr:
        for p, e in (nerr + cerr):
            print(f"[ERROR] {p}: {e}")
    print(f"[INFO] found noisy={len(ninfo)} wavs, clean={len(cinfo)} wavs")

    # Basic audio checks
    def stats(info):
        bad_sr = [p for (p,sr,ch,d) in info if sr not in (16000,)]
        bad_ch = [p for (p,sr,ch,d) in info if ch not in (1,)]
        short  = [p for (p,sr,ch,d) in info if d <= 0.0]
        return bad_sr, bad_ch, short

    nb_sr, nb_ch, nb_short = stats(ninfo)
    cb_sr, cb_ch, cb_short = stats(cinfo)

    if nb_sr or cb_sr: print(f"[WARN] sample-rate != 16k: noisy:{len(nb_sr)} clean:{len(cb_sr)}")
    if nb_ch or cb_ch: print(f"[WARN] channels != mono:  noisy:{len(nb_ch)} clean:{len(cb_ch)}")
    if nb_short or cb_short: print(f"[WARN] zero/short durations: noisy:{len(nb_short)} clean:{len(cb_short)}")

    # Pairing
    pairs, miss_n, miss_c = pairs_report(noisy_dir, clean_dir)
    print(f"[INFO] paired {len(pairs)} files; missing_noisy={len(miss_n)} missing_clean={len(miss_c)}")
    if miss_n: print(f"[WARN] clean-without-noisy:  e.g. {miss_n[:5]}")
    if miss_c: print(f"[WARN] noisy-without-clean: e.g. {miss_c[:5]}")

    # Write manifest
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(pairs)
    print(f"[OK] wrote manifest: {out_csv} (pairs={len(pairs)})")

    # Hard fail if no pairs
    if not pairs:
        print(f"[ERROR] no pairs in {name} — verifică structura directoarelor", file=sys.stderr)
        sys.exit(2)

VBD = os.environ["VBD_DIR"]
train_csv = os.path.join(VBD, "train.csv")
test_csv  = os.path.join(VBD, "test.csv")

check_split("VOICEBANK-TRAIN",
            os.environ["TRAIN_NOISY"],
            os.environ["TRAIN_CLEAN"],
            train_csv)

check_split("VOICEBANK-TEST",
            os.environ["TEST_NOISY"],
            os.environ["TEST_CLEAN"],
            test_csv)

# LibriSpeech sanity
LIBRI = os.environ.get("LIBRI_DIR", "")
if LIBRI and os.path.isdir(LIBRI):
    header("[CHECK] LibriSpeech subsets")
    subsets = ["train-clean-100","train-clean-360","dev-clean","test-clean","train-other-500","dev-other","test-other"]
    total = 0
    for sub in subsets:
        d = os.path.join(LIBRI, "LibriSpeech", sub)
        if os.path.isdir(d):
            flacs = glob.glob(os.path.join(d, "**", "*.flac"), recursive=True)
            total += len(flacs)
            print(f"[INFO] {sub}: {len(flacs)} flac")
    lst = os.path.join(LIBRI, "clean_files.lst")
    with open(lst, "w") as f:
        for p in glob.glob(os.path.join(LIBRI, "LibriSpeech", "**", "*.flac"), recursive=True):
            f.write(p+"\n")
    print(f"[OK] wrote {lst} (total={total})")

print("\n[ALL GOOD] VoiceBank manifests + LibriSpeech checks complete.")
PY

echo
echo "[SUMMARY]"
echo "  VoiceBank-DEMAND:"
echo "    - manifest train : ${VBD_DIR}/train.csv"
echo "    - manifest test  : ${VBD_DIR}/test.csv"
echo "  LibriSpeech:"
echo "    - list clean flac: ${LIBRI_DIR}/clean_files.lst (dacă LibriSpeech e instalat)"
echo
echo "[DONE] Integrity checked & manifests ready."
