#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/get_librispeech.sh <LIBRISPEECH_ROOT>
#
# Env:
#   LIBRISPEECH_SUBSETS="train-clean-100 dev-clean test-clean"
#   FORCE=1        # re-extragere/descărcare dacă lipsesc .flac-urile
#
# Recomandare: setează în .env.local
#   LIBRISPEECH_ROOT=$(PWD)/data/librispeech
# (NU adăuga /LibriSpeech la final — arhivele creează acel folder uneori)

ROOT="${1:-}"
if [[ -z "${ROOT}" ]]; then
  echo "[get_librispeech] Usage: $0 <LIBRISPEECH_ROOT>"
  exit 2
fi

command -v wget >/dev/null 2>&1 || { echo "[get_librispeech] ERROR: wget not found"; exit 2; }
mkdir -p "${ROOT}"
cd "${ROOT}"

LIBRISPEECH_SUBSETS="${LIBRISPEECH_SUBSETS:-train-clean-100 dev-clean}"
BASE_URL="https://www.openslr.org/resources/12"

log(){ printf '%s\n' "$*" >&2; }

has_flac_anywhere() {
  find . -type f -name '*.flac' -print -quit | grep -q .
}

has_flac_subset() {
  local subset="$1"
  # Acceptă atât layout-ul cu wrapper (LibriSpeech/subset) cât și fără (subset)
  { find "${subset}" -type f -name '*.flac' 2>/dev/null || true; \
    find "LibriSpeech/${subset}" -type f -name '*.flac' 2>/dev/null || true; } | grep -q .
}

flatten_wrapper_once() {
  # Dacă extragerea a creat ./LibriSpeech/<subset>/..., mută conținutul în ROOT/
  if [[ -d "LibriSpeech" ]]; then
    shopt -s dotglob nullglob
    local moved=0
    for f in LibriSpeech/*; do
      # nu suprascrie dacă deja există (mv -n)
      mv -n "$f" . 2>/dev/null || true
      moved=1
    done
    shopt -u dotglob nullglob
    # încerci să elimini wrapper-ul dacă e gol
    rmdir LibriSpeech 2>/dev/null || true
    if [[ "${moved}" -eq 1 ]]; then
      log "[get_librispeech] flattened 'LibriSpeech/' wrapper into ${ROOT}"
    fi
  fi
}

download_and_extract () {
  local subset="$1"
  local tgz="${subset}.tar.gz"

  # Dacă subsetul are deja .flac, sari (cu excepția cazului FORCE=1)
  if has_flac_subset "${subset}" && [[ -z "${FORCE:-}" ]]; then
    log "[get_librispeech] '${subset}' already has .flac → skip"
    return 0
  fi

  # Dacă arhiva e locală, încearcă (re)extragerea din cache mai întâi
  if [[ -f "${tgz}" ]]; then
    log "[get_librispeech] extracting cached ${tgz} ..."
    tar -xzf "${tgz}"
  else
    log "[get_librispeech] fetching ${tgz} ..."
    wget -c --no-verbose "${BASE_URL}/${tgz}"
    log "[get_librispeech] extracting ${tgz} ..."
    tar -xzf "${tgz}"
  fi

  # După extragere, aplatizează dacă e cazul
  flatten_wrapper_once

  # Validare subset după extragere; dacă nu există .flac, forțează re-descarcarea
  if ! has_flac_subset "${subset}"; then
    log "[get_librispeech] WARN: no .flac for subset '${subset}' after extract; re-fetching…"
    rm -f "${tgz}"
    wget -c --no-verbose "${BASE_URL}/${tgz}"
    tar -xzf "${tgz}"
    flatten_wrapper_once
  fi
}

for s in ${LIBRISPEECH_SUBSETS}; do
  download_and_extract "${s}"
done

# Validare finală: trebuie să existe cel puțin un .flac sub ROOT
if ! has_flac_anywhere; then
  echo "[get_librispeech] ERROR: no .flac found under ${ROOT}"
  exit 1
fi

echo "[get_librispeech] OK under ${ROOT}"
