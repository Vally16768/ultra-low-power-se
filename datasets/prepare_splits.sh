#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CLI
# =========================================================
DATASETS=${1:-all}   # all | vb | dns3 | dns4 | dns5 (spațiu-separate ex. "vb dns5")
shift || true

IN_ROOT="./datasets"                     # unde ai rulat download_data.sh
OUT_ROOT="$IN_ROOT/prepared"
VB_SR="16k"                              # 16k sau 48k (în funcție de ce ai pregătit)
VB_VAL_MODE="none"                       # none | alias-test
DNS5_TRACK="headset"                     # headset | speakerphone
DNS_VAL_SRC="dev"                        # dev | none
DNS_TEST_SRC="none"                      # dev | blind | none
LINK_MODE="symlink"                      # symlink | copy

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in)           IN_ROOT="$2"; shift 2 ;;
    --out)          OUT_ROOT="$2"; shift 2 ;;
    --vb-sr)        VB_SR="$2"; shift 2 ;;
    --vb-val)       VB_VAL_MODE="$2"; shift 2 ;;
    --dns5-track)   DNS5_TRACK="$2"; shift 2 ;;
    --dns-val-src)  DNS_VAL_SRC="$2"; shift 2 ;;
    --dns-test-src) DNS_TEST_SRC="$2"; shift 2 ;;
    --link)         LINK_MODE="$2"; shift 2 ;;
    *) echo "Arg necunoscut: $1" >&2; exit 1 ;;
  esac
done

# =========================================================
# Utils
# =========================================================
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Instalează $1" >&2; exit 1; }; }
need_cmd find
need_cmd awk
need_cmd sed
need_cmd python3

ln_or_cp() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ "$LINK_MODE" == "copy" ]]; then
    cp -n "$src" "$dst"
  else
    ln -sfn "$(realpath "$src")" "$dst"
  fi
}

write_pairs_csv_py='
import csv, os, sys
pairs = []
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if f.lower().endswith(".wav"):
            pairs.append(os.path.join(root,f))
clean = [p for p in pairs if "/clean/" in p or p.lower().find("clean")!=-1]
noisy = [p for p in pairs if "/noisy/" in p or p.lower().find("noisy")!=-1]
# Map by basename:
clean_map = {os.path.splitext(os.path.basename(p))[0]: p for p in clean}
rows = []
for n in noisy:
    k = os.path.splitext(os.path.basename(n))[0]
    if k in clean_map:
        rows.append((n, clean_map[k]))
if not rows:
    print("[WARN] zero pairs in", sys.argv[1], file=sys.stderr)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["noisy","clean"])
    w.writerows(rows)
'

make_split() {
  local name="$1" split="$2" src_noisy="$3" src_clean="$4" out_dir="$OUT_ROOT/$name/$split"
  [[ -d "$src_noisy" && -d "$src_clean" ]] || { echo "[SKIP] $name:$split nu găsesc $src_noisy / $src_clean"; return 0; }
  echo ">> $name:$split -> $out_dir"
  mkdir -p "$out_dir/noisy" "$out_dir/clean"
  find "$src_noisy" -type f -iname '*.wav' | while read -r f; do
    rel="${f#$src_noisy/}"
    ln_or_cp "$f" "$out_dir/noisy/$rel"
  done
  find "$src_clean" -type f -iname '*.wav' | while read -r f; do
    rel="${f#$src_clean/}"
    ln_or_cp "$f" "$out_dir/clean/$rel"
  done
  python3 - <<PY "$out_dir" "$out_dir/pairs_${split}.csv"
$write_pairs_csv_py
PY
}

# Heuristici mici pentru a găsi folderele (rezistente la layouturile oficiale).
guess_dir() {
  # $1 root, $2 regex to match path (awk-compatible, case-insens)
  find "$1" -type d | awk -v pat="$2" 'BEGIN{IGNORECASE=1} $0 ~ pat {print}' | head -n1
}

# =========================================================
# VoiceBank+DEMAND (oficial: train 28spk / test 2spk; fără val)
# =========================================================
prep_vb() {
  local base="$IN_ROOT/voicebank-demand/$VB_SR"
  local noisy_train="$base/noisy_train"
  local clean_train="$base/clean_train"
  local noisy_test="$base/noisy_test"
  local clean_test="$base/clean_test"
  [[ -d "$base" ]] || { echo "[ERR] Nu găsesc $base (rulează download_data.sh vb --vb-16k dacă vrei 16k)"; return 1; }

  make_split "voicebank-demand/$VB_SR" "train" "$noisy_train" "$clean_train"
  make_split "voicebank-demand/$VB_SR" "test"  "$noisy_test"  "$clean_test"

  if [[ "$VB_VAL_MODE" == "alias-test" ]]; then
    echo ">> VB val = alias la test (NU folosi pentru scorul final!)."
    # creează doar link către același conținut
    local out_val="$OUT_ROOT/voicebank-demand/$VB_SR/val"
    mkdir -p "$(dirname "$out_val")"
    rm -rf "$out_val"
    ln -sfn "../test" "$out_val"
  else
    echo ">> VB nu are val oficial -> omit val."
  fi
}

# =========================================================
# DNS v3/v4/v5 (folosim partițiile oficiale dacă există local)
# - train: training set oficial
# - val:   dev set (dacă există perechi)
# - test:  dev (cu perechi) sau blind (fără clean) sau none (implicit)
# =========================================================
prep_dns() {
  local ver="$1"  # dns3 | dns4 | dns5
  local dns_root="$IN_ROOT/dns_challenge"

  [[ -d "$dns_root" ]] || { echo "[ERR] Nu găsesc $dns_root. Rulează download-urile Microsoft."; return 1; }

  # În multe setup-uri, scripturile Microsoft creează foldere cu numele versiunii.
  # Căutăm euristic:
  local base=$(guess_dir "$dns_root" "$ver")
  [[ -n "$base" ]] || { echo "[WARN] Nu găsesc director pentru $ver sub $dns_root. Sar peste."; return 0; }

  # TRAIN (căutăm directoare care conțin "train" + noisy/clean)
  local tr_noisy=$(guess_dir "$base" "train.*noisy|noisy.*train")
  local tr_clean=$(guess_dir "$base" "train.*clean|clean.*train")

  # DEV/VAL
  local dev_noisy=$(guess_dir "$base" "dev.*(test)?set.*noisy|noisy.*dev")
  local dev_clean=$(guess_dir "$base" "dev.*(test)?set.*clean|clean.*dev")

  # BLIND TEST (de obicei fără clean)
  local blind_noisy=$(guess_dir "$base" "blind.*test.*noisy|noisy.*blind")
  local blind_clean=""  # nu există în mod normal

  # «dns5-track» afectează doar structurile unde există headset/speakerphone;
  # dacă găsim subfoldere cu denumirea track-ului, restrângem.
  if [[ "$ver" == "dns5" ]]; then
    local track_dir=$(guess_dir "$base" "$DNS5_TRACK")
    if [[ -n "$track_dir" ]]; then
      base="$track_dir"
      tr_noisy=$(guess_dir "$base" "train.*noisy|noisy.*train")
      tr_clean=$(guess_dir "$base" "train.*clean|clean.*train")
      dev_noisy=$(guess_dir "$base" "dev.*(test)?set.*noisy|noisy.*dev")
      dev_clean=$(guess_dir "$base" "dev.*(test)?set.*clean|clean.*dev")
      blind_noisy=$(guess_dir "$base" "blind.*test.*noisy|noisy.*blind")
    fi
  fi

  # Train
  if [[ -n "$tr_noisy" && -n "$tr_clean" ]]; then
    make_split "$ver" "train" "$tr_noisy" "$tr_clean"
  else
    echo "[WARN] $ver: nu am găsit trenul oficial (noisy/clean)."
  fi

  # Val
  if [[ "$DNS_VAL_SRC" == "dev" && -n "$dev_noisy" && -n "$dev_clean" ]]; then
    make_split "$ver" "val" "$dev_noisy" "$dev_clean"
  else
    echo ">> $ver: omit val (DNS_VAL_SRC=$DNS_VAL_SRC sau dev lipsă)."
  fi

  # Test
  case "$DNS_TEST_SRC" in
    dev)
      if [[ -n "$dev_noisy" && -n "$dev_clean" ]]; then
        make_split "$ver" "test" "$dev_noisy" "$dev_clean"
      else
        echo "[WARN] $ver: cerut test=dev, dar dev nu e găsit."
      fi
      ;;
    blind)
      if [[ -n "$blind_noisy" ]]; then
        # facem doar noisy; pairs.csv va avea 0 rânduri (fără clean)
        local out_dir="$OUT_ROOT/$ver/test"
        mkdir -p "$out_dir/noisy" "$out_dir/clean"
        find "$blind_noisy" -type f -iname '*.wav' | while read -r f; do
          rel="${f#$blind_noisy/}"; ln_or_cp "$f" "$out_dir/noisy/$rel"
        done
        # CSV gol cu header
        echo "noisy,clean" > "$out_dir/pairs_test.csv"
      else
        echo "[WARN] $ver: cerut test=blind, dar blind nu e găsit."
      fi
      ;;
    none) echo ">> $ver: omit test (DNS_TEST_SRC=none)";;
    *) echo "[WARN] $ver: DNS_TEST_SRC necunoscut: $DNS_TEST_SRC";;
  esac
}

# =========================================================
# Dispatch
# =========================================================
echo "IN_ROOT=$IN_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
mkdir -p "$OUT_ROOT"

case "$DATASETS" in
  all)   prep_vb; prep_dns dns3; prep_dns dns4; prep_dns dns5 ;;
  vb)    prep_vb ;;
  dns3)  prep_dns dns3 ;;
  dns4)  prep_dns dns4 ;;
  dns5)  prep_dns dns5 ;;
  *)
    for d in $DATASETS; do
      case "$d" in
        vb) prep_vb ;;
        dns3|dns4|dns5) prep_dns "$d" ;;
        *) echo "Dataset necunoscut: $d" >&2; exit 1 ;;
      esac
    done
  ;;
esac

echo ">> Gata. Structurile standard sunt în: $OUT_ROOT"
