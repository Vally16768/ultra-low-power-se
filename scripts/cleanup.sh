#!/usr/bin/env bash
# Cleanup residual training artifacts for Ultra-Low-Power Speech Enhancement
# Usage:
#   scripts/cleanup.sh                 # dry-run (nu șterge nimic)
#   scripts/cleanup.sh --yes           # chiar șterge (presetul "basic")
#   scripts/cleanup.sh --aggressive    # șterge mai mult (vezi listele) - dry-run
#   scripts/cleanup.sh --aggressive --yes
#   scripts/cleanup.sh --only tb,wandb,artifacts # selectiv
#   scripts/cleanup.sh --help

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARTIFACTS_DIR="artifacts"
# Subfoldere tipice din artifacts (din screenshot/executabilele proiectului)
ARTIFACTS_SUBDIRS=(
  "$ARTIFACTS_DIR/exp"
  "$ARTIFACTS_DIR/infer"
  "$ARTIFACTS_DIR/runs"
  "$ARTIFACTS_DIR/score"
  "$ARTIFACTS_DIR/tmp"
  "$ARTIFACTS_DIR/checkpoints"
  "$ARTIFACTS_DIR/quant"
  "$ARTIFACTS_DIR/profiling"
  "$ARTIFACTS_DIR/metrics"
  "$ARTIFACTS_DIR/export/*.tmp"
)

# Paletă liste (preseturi)
BASIC_DIRS=(
  "${ARTIFACTS_SUBDIRS[@]}"
  "results"
  "logs"
  "runs"                 # TensorBoard default
  "lightning_logs"       # PyTorch Lightning
  "wandb"
  "mlruns"               # MLflow
  ".pytest_cache"
  ".ruff_cache"
  ".mypy_cache"
  ".cache/torch/hub"
  ".cache/huggingface/datasets"
  "data/prepared/**/tmp"
  "tmp"
)
BASIC_FILES=(
  "$ARTIFACTS_DIR/export/*.onnx.bak"
  "$ARTIFACTS_DIR/export/*.json.tmp"
  "$ARTIFACTS_DIR/*.tmp"
  "$ARTIFACTS_DIR/*.log"
  "artifacts/*.log"
  "*.ckpt.tmp"
)

# Aggressive adaugă și:
AGGR_DIRS_EXTRA=(
  "$ARTIFACTS_DIR"       # include TOT artifacts (filtrăm păstrările în modul basic)
  "dist" "build"         # roți / build cache
  "__pycache__"
  ".ipynb_checkpoints"
  "data/prepared/**/mixes"  # dataseturi sintetizate regenerabile
  "tensorboard"          # alte aliasuri locale
)
AGGR_FILES_EXTRA=(
  "*.ckpt"
  "*.pt"
  "*.pth"
  "$ARTIFACTS_DIR/export/*.onnx"   # ATENȚIE: șters doar în modul --aggressive
)

# Set de componente adresabile cu --only
declare -A NAMED_SETS=(
  # cheie -> "dirs;files" (liste separate prin |)
  [tb]="runs|lightning_logs|tensorboard ; "
  [wandb]="wandb ; "
  [mlflow]="mlruns ; "
  [checkpoints]="$ARTIFACTS_DIR/checkpoints|*.ckpt|*.pt|*.pth|*.ckpt.tmp ; "
  [onnx_tmp]="$ARTIFACTS_DIR/export/*.tmp|$ARTIFACTS_DIR/export/*.json.tmp|$ARTIFACTS_DIR/export/*.onnx.bak ; "
  [results]="results|$ARTIFACTS_DIR/metrics ; "
  [caches]=".pytest_cache|.ruff_cache|.mypy_cache|__pycache__|.ipynb_checkpoints|.cache/torch/hub|.cache/huggingface/datasets ; "
  [tmp]="tmp|data/prepared/**/tmp ; "
  # NOU: curățare țintită pentru artifacts (EXCLUS rădăcina cu fișiere utile în basic)
  [artifacts]="$ARTIFACTS_DIR/exp|$ARTIFACTS_DIR/infer|$ARTIFACTS_DIR/runs|$ARTIFACTS_DIR/score|$ARTIFACTS_DIR/tmp|$ARTIFACTS_DIR/checkpoints|$ARTIFACTS_DIR/quant|$ARTIFACTS_DIR/profiling|$ARTIFACTS_DIR/metrics ; $ARTIFACTS_DIR/*.tmp|$ARTIFACTS_DIR/*.log"
)

DRY_RUN=1
AGGRESSIVE=0
ONLY=""
YES=0

print_help() {
  cat <<EOF
Cleanup script (sigur, cu dry-run implicit).
Opțiuni:
  --yes            Execută ștergerile (altfel doar afișează ce ar face).
  --aggressive     Include fișiere/directoare suplimentare (vezi script).
  --only X[,Y...]  Curăță doar componentele numite: $(printf "%s " "${!NAMED_SETS[@]}").
  --root DIR       Rădăcina proiectului (default: $ROOT_DIR)
  --help           Arată acest mesaj.
Exemple:
  scripts/cleanup.sh                     # dry-run basic
  scripts/cleanup.sh --yes               # cleanup basic
  scripts/cleanup.sh --aggressive --yes  # cleanup complet (poate goli și artifacts/)
  scripts/cleanup.sh --only artifacts --yes  # doar artifacts/* (păstrează fișierele utile din rădăcină)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=1; DRY_RUN=0; shift ;;
    --aggressive) AGGRESSIVE=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --root) ROOT_DIR="${2:-}"; shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Opțiune necunoscută: $1"; print_help; exit 1 ;;
  esac
done

banner() { echo -e "\033[1;36m$*\033[0m"; }
warn()   { echo -e "\033[1;33m$*\033[0m"; }
err()    { echo -e "\033[1;31m$*\033[0m" >&2; }

cd "$ROOT_DIR"

# Protecție: nu șterge foldere critice
PROTECT_DIRS=("data/raw" "configs" ".git" "scripts")
for p in "${PROTECT_DIRS[@]}"; do
  [[ -e "$p" ]] || true
done

# Construiește listele țintă
DIRS=("${BASIC_DIRS[@]}")
FILES=("${BASIC_FILES[@]}")
if [[ $AGGRESSIVE -eq 1 ]]; then
  DIRS+=("${AGGR_DIRS_EXTRA[@]}")
  FILES+=("${AGGR_FILES_EXTRA[@]}")
fi

# Override prin --only
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a keys <<< "$ONLY"
  DIRS=(); FILES=()
  for k in "${keys[@]}"; do
    if [[ -n "${NAMED_SETS[$k]:-}" ]]; then
      IFS=';' read -r dirs_part files_part <<< "${NAMED_SETS[$k]}"
      IFS='|' read -r -a dlist <<< "$(echo "$dirs_part" | xargs)"
      for d in "${dlist[@]}"; do [[ -n "$d" ]] && DIRS+=("$d"); done
      IFS='|' read -r -a flist <<< "$(echo "$files_part" | xargs)"
      for f in "${flist[@]}"; do [[ -n "$f" ]] && FILES+=("$f"); done
    else
      warn "Ignor set necunoscut: $k"
    fi
  done
fi

banner "Root: $ROOT_DIR"
[[ $DRY_RUN -eq 1 ]] && banner "MOD: DRY-RUN (nu se șterge nimic)" || banner "MOD: EXECUȚIE (se șterge)"

# Colectează candidaturi
TO_DELETE=()

collect_glob() {
  local pattern="$1"
  shopt -s nullglob globstar
  for path in $pattern; do
    [[ -e "$path" ]] || continue
    # Protecție: nu intra în dir-uri critice
    for prot in "${PROTECT_DIRS[@]}"; do
      [[ "$path" == "$prot"* ]] && { warn "Protejat, sar: $path"; continue 2; }
    done
    TO_DELETE+=("$path")
  done
  shopt -u nullglob globstar
}

for d in "${DIRS[@]}";   do collect_glob "$d"; done
for f in "${FILES[@]}";  do collect_glob "$f"; done

# Păstrări în modul BASIC: nu șterge fișiere utile din rădăcina artifacts/
# (ex: model_int8.onnx, report.html, bench_quick.json, demo_out.wav)
KEEP_GLOBS_BASIC=(
  "$ARTIFACTS_DIR/*.onnx"
  "$ARTIFACTS_DIR/*.html"
  "$ARTIFACTS_DIR/*.json"
  "$ARTIFACTS_DIR/*.wav"
)

should_keep_basic() {
  local p="$1"
  for g in "${KEEP_GLOBS_BASIC[@]}"; do
    shopt -s nullglob
    for k in $g; do
      [[ "$p" == "$k" ]] && { shopt -u nullglob; return 0; }
    done
    shopt -u nullglob
  done
  return 1
}

# În modul basic, nu șterge .onnx și alte fișiere-cheie din artifacts/
if [[ $AGGRESSIVE -eq 0 ]]; then
  filtered=()
  for p in "${TO_DELETE[@]}"; do
    if should_keep_basic "$p"; then
      warn "Păstrez (basic mode): $p"
      continue
    fi
    filtered+=("$p")
  done
  TO_DELETE=("${filtered[@]}")
fi

# Unicizează (fără duplicate)
mapfile -t TO_DELETE < <(printf "%s\n" "${TO_DELETE[@]}" | awk '!seen[$0]++')

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
  banner "Nimic de curățat conform presetului."
  exit 0
fi

echo "Ținte găsite (${#TO_DELETE[@]}):"
for p in "${TO_DELETE[@]}"; do
  echo "  - $p"
done

if [[ $DRY_RUN -eq 1 ]]; then
  banner "Dry-run terminat."
  exit 0
fi

# Confirmare dacă nu s-a trecut --yes (redundanță)
if [[ $YES -ne 1 ]]; then
  read -r -p "Confirmi ștergerea? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Abandon."; exit 1; }
fi

# Mută în .trash cu timestamp (siguranță), nu șterge direct
TS="$(date +%Y%m%d_%H%M%S)"
TRASH=".trash/$TS"
mkdir -p "$TRASH"

banner "Mut fișierele în $TRASH (poți șterge .trash manual ulterior)"
for p in "${TO_DELETE[@]}"; do
  rel="${p#./}"
  rel="${rel#$ROOT_DIR/}"
  dest="$TRASH/$rel"
  mkdir -p "$(dirname "$dest")"
  mv "$p" "$dest" 2>/dev/null || {
    [[ -d "$p" ]] && { mkdir -p "$dest"; rmdir "$p" 2>/dev/null || true; }
  }
done

banner "Gata. Artefactele au fost mutate în: $TRASH"
echo "Poți rula: rm -rf .trash după ce verifici."
