#!/usr/bin/env bash
set -euo pipefail

ROOT="data"
LIBRI="$ROOT/clean_wav/libri"
DEMAND="$ROOT/noise/demand"
NOISES_CUSTOM="$ROOT/noise/custom"
NOISES_DEV="$ROOT/noise/dev"
NOISES_UNSEEN="$ROOT/noise/unseen"
NOISES_SYNTH="$ROOT/noise/synth"   # <<— nou: sintetice
RIRS="$ROOT/rirs"

mkdir -p "$ROOT/lists"

# ---------- CLEAN ----------
# train: train-clean-* dacă există, altfel dev-other ca fallback
if compgen -G "$LIBRI/train-clean-*/**/*.wav" > /dev/null; then
  find "$LIBRI" -type f -path "$LIBRI/train-clean-*/*" -iname "*.wav" | sort > "$ROOT/lists/train_clean.txt"
else
  if [ -d "$LIBRI/dev-other" ]; then
    find "$LIBRI/dev-other" -type f -iname "*.wav" | sort > "$ROOT/lists/train_clean.txt"
  else
    echo "Eroare: nu găsesc $LIBRI/train-clean-* sau $LIBRI/dev-other pentru train." >&2
    exit 2
  fi
fi

# dev: dev-clean obligatoriu
if [ -d "$LIBRI/dev-clean" ]; then
  find "$LIBRI/dev-clean" -type f -iname "*.wav" | sort > "$ROOT/lists/dev_clean.txt"
else
  echo "Eroare: nu găsesc $LIBRI/dev-clean." >&2
  exit 2
fi

# ---------- NOISES ----------
# train noises: DEMAND + custom + synth (oricare există)
: > "$ROOT/lists/noise_train.txt"
for D in "$DEMAND" "$NOISES_CUSTOM" "$NOISES_SYNTH"; do
  if [ -d "$D" ]; then
    find "$D" -type f -iname "*.wav" | sort >> "$ROOT/lists/noise_train.txt"
  fi
done
if [ ! -s "$ROOT/lists/noise_train.txt" ]; then
  echo "Eroare: nu am găsit zgomote pentru train în $DEMAND / $NOISES_CUSTOM / $NOISES_SYNTH" >&2
  exit 2
fi

# dev noises: dacă ai director dedicat, altfel ia primele 200 din train
if [ -d "$NOISES_DEV" ]; then
  find "$NOISES_DEV" -type f -iname "*.wav" | sort > "$ROOT/lists/noise_dev.txt"
else
  head -n 200 "$ROOT/lists/noise_train.txt" > "$ROOT/lists/noise_dev.txt"
fi

# unseen noises: preferă director dedicat; altfel 50 random din train care nu sunt în dev
if [ -d "$NOISES_UNSEEN" ]; then
  find "$NOISES_UNSEEN" -type f -iname "*.wav" | sort > "$ROOT/lists/noise_unseen.txt"
else
  comm -23 <(sort "$ROOT/lists/noise_train.txt") <(sort "$ROOT/lists/noise_dev.txt") | shuf -n 50 > "$ROOT/lists/noise_unseen.txt"
  # dacă tot a rămas gol (puține fișiere), ia 50 random din train
  if [ ! -s "$ROOT/lists/noise_unseen.txt" ]; then
    shuf "$ROOT/lists/noise_train.txt" | head -n 50 > "$ROOT/lists/noise_unseen.txt"
  fi
fi

# ---------- RIRs (opțional) ----------
if [ -d "$RIRS" ]; then
  find "$RIRS" -type f -iname "*.wav" | sort > "$ROOT/lists/rir_list.txt" || true
fi

# ---------- Rezumat ----------
for f in train_clean.txt dev_clean.txt noise_train.txt noise_dev.txt noise_unseen.txt rir_list.txt; do
  [ -f "$ROOT/lists/$f" ] && printf "%7d  %s\n" "$(wc -l < "$ROOT/lists/$f")" "$ROOT/lists/$f"
done
