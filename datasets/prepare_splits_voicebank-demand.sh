#!/usr/bin/env bash
# prepare_splits_voicebank-demand.sh
# Build CSVs with (noisy,clean) pairs for VoiceBank-DEMAND (16k) layout:
# datasets/voicebank-demand/16k/{clean_train,noisy_train,clean_test,noisy_test}
set -euo pipefail

# --- Config / Paths -----------------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
DATA_DIR="${SCRIPT_DIR}/datasets/voicebank-demand/16k"

CLEAN_TRAIN="${DATA_DIR}/clean_train"
NOISY_TRAIN="${DATA_DIR}/noisy_train"
CLEAN_TEST="${DATA_DIR}/clean_test"
NOISY_TEST="${DATA_DIR}/noisy_test"

OUT_TRAIN="${DATA_DIR}/train.csv"
OUT_TEST="${DATA_DIR}/test.csv"

# --- Helpers ------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

check_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

pair_split() {
  local noisy_dir="$1"
  local clean_dir="$2"
  local out_csv="$3"

  # Create a temp dir that cleans up on exit
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf -- "'"$tmp"'"' RETURN

  local noisy_list="${tmp}/noisy.lst"
  local clean_list="${tmp}/clean.lst"
  local noisy_keys="${tmp}/noisy_keys.tsv"
  local clean_keys="${tmp}/clean_keys.tsv"
  local joined="${tmp}/joined.tsv"

  # Collect files (null-delimited, safe for odd names), sort by basename
  find "$noisy_dir" -type f -iname '*.wav' -print0 \
    | xargs -0 -I{} bash -c 'f="{}"; b="$(basename "${f%.*}")"; printf "%s\t%s\n" "$b" "$f"' \
    | sort -k1,1 > "$noisy_keys"

  find "$clean_dir" -type f -iname '*.wav' -print0 \
    | xargs -0 -I{} bash -c 'f="{}"; b="$(basename "${f%.*}")"; printf "%s\t%s\n" "$b" "$f"' \
    | sort -k1,1 > "$clean_keys"

  # Report simple stats
  local n_noisy n_clean
  n_noisy=$(wc -l < "$noisy_keys" | tr -d ' ')
  n_clean=$(wc -l < "$clean_keys" | tr -d ' ')
  echo "Found: $n_noisy noisy files in $noisy_dir"
  echo "Found: $n_clean clean files in $clean_dir"

  # Join by basename (left-join from noisy -> clean) to create aligned pairs
  # Output format: noisy_path,clean_path
  join -t $'\t' -j 1 -o 1.2,2.2 "$noisy_keys" "$clean_keys" > "$joined" || true

  local n_joined
  n_joined=$(wc -l < "$joined" | tr -d ' ')
  echo "Paired: $n_joined files"

  # Warn if there are mismatches
  if (( n_joined == 0 )); then
    die "No pairs matched between $(basename "$noisy_dir") and $(basename "$clean_dir"). Check filenames."
  fi
  if (( n_joined < n_noisy )); then
    echo "WARNING: $(("$n_noisy" - "$n_joined")) noisy files had no matching clean partner." >&2
    # Show a few missing for quick debugging
    comm -23 <(cut -f1 "$noisy_keys") <(cut -f1 "$clean_keys") | head -n 10 \
      | sed 's/^/  missing clean for basename: /' >&2
  fi
  if (( n_joined < n_clean )); then
    echo "NOTE: $(("$n_clean" - "$n_joined")) clean files had no matching noisy partner." >&2
  fi

  # Write CSV header + rows; store paths relative to repo root for portability
  {
    echo "noisy,clean"
    awk -F'\t' -v root="$SCRIPT_DIR/" '{
      # Convert absolute to relative if under repo; otherwise keep as-is
      noisy=$1; clean=$2;
      sub("^"root, "", noisy); sub("^"root, "", clean);
      gsub(/\\/, "/", noisy); gsub(/\\/, "/", clean); # normalize slashes
      printf "%s,%s\n", noisy, clean
    }' "$joined"
  } > "$out_csv"

  echo "Wrote: $out_csv"
}

# --- Checks -------------------------------------------------------------------
check_dir "$DATA_DIR"
check_dir "$CLEAN_TRAIN"
check_dir "$NOISY_TRAIN"
check_dir "$CLEAN_TEST"
check_dir "$NOISY_TEST"

# --- Build CSVs ---------------------------------------------------------------
echo "Building VoiceBank-DEMAND (16k) splits…"
pair_split "$NOISY_TRAIN" "$CLEAN_TRAIN" "$OUT_TRAIN"
pair_split "$NOISY_TEST"  "$CLEAN_TEST"  "$OUT_TEST"

echo "Done."
