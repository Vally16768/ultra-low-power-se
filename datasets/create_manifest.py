#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_manifest.py
Generează două manifeste CSV:
  - train.csv: combinație de perechi VoiceBank (pre-paired) + fișiere LibriSpeech (pentru augmentare)
  - val.csv: numai perechi VoiceBank (pre-paired) pentru evaluare consistentă

Coloane CSV:
  clean_path,noisy_path,type

where:
  type == "pre-paired"  -> noisy_path este fișierul existent (VoiceBank)
  type == "augment"     -> noisy_path == "N/A" (va fi generat la încărcare în dataset)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------
# CONFIGUREAZĂ CĂILE AICI
# ----------------------------------------------------
# Calea către folderul rădăcină al datelor
DATA_ROOT = Path("/home/vpopescu/projects/ultra-low-power-se/data")

# Căile către Voicebank-DEMAND
VOICEBANK_ROOT = DATA_ROOT / "voicebank-demand-16k"
VB_CLEAN_TRAIN = VOICEBANK_ROOT / "clean_trainset_28spk_wav"
VB_NOISY_TRAIN = VOICEBANK_ROOT / "noisy_trainset_28spk_wav"
VB_CLEAN_VAL   = VOICEBANK_ROOT / "clean_testset_wav"
VB_NOISY_VAL   = VOICEBANK_ROOT / "noisy_testset_wav"

# Căile către Librispeech (doar audio curat)
LIBRISPEECH_ROOT      = DATA_ROOT / "librispeech" / "LibriSpeech"
LS_CLEAN_TRAIN_100    = LIBRISPEECH_ROOT / "train-clean-100"
LS_CLEAN_TRAIN_360    = LIBRISPEECH_ROOT / "train-clean-360"
LS_CLEAN_VAL          = LIBRISPEECH_ROOT / "dev-clean"  # (nefolosit în val în acest script)

# Căile de ieșire pentru manifest
MANIFEST_DIR   = Path("/home/vpopescu/projects/ultra-low-power-se/manifests")
TRAIN_CSV_PATH = MANIFEST_DIR / "train.csv"
VAL_CSV_PATH   = MANIFEST_DIR / "val.csv"
# ----------------------------------------------------


def _require_dir(path: Path, name: str) -> None:
    """Asigură că directorul există (altfel oprește scriptul cu un mesaj clar)."""
    if not path.exists() or not path.is_dir():
        print(f"[E] Directorul {name} nu există: {path}", file=sys.stderr)
        sys.exit(1)


def find_files(path: Path, extension: str = ".wav") -> List[Path]:
    """Scanează recursiv un folder după fișiere cu extensia dată."""
    return sorted(path.rglob(f"*{extension}"))


def process_voicebank(clean_dir: Path, noisy_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Generează perechi pre-existente din VoiceBank:
      (clean_path, noisy_path, "pre-paired")

    Perechile se asociază după numele fișierului (ex: p232_001.wav).
    """
    items: List[Tuple[str, str, str]] = []
    _require_dir(clean_dir, "VB clean")
    _require_dir(noisy_dir, "VB noisy")

    noisy_files = find_files(noisy_dir, extension=".wav")
    print(f"[Info] Procesare VoiceBank: {clean_dir.name} vs {noisy_dir.name}")
    if not noisy_files:
        print(f"[W] Nu am găsit fișiere în: {noisy_dir}", file=sys.stderr)

    # Mapare rapidă: clean_path = clean_dir / noisy_file.name
    for noisy_path in tqdm(noisy_files, desc="  Scanare VoiceBank", unit="file"):
        clean_path = clean_dir / noisy_path.name
        if clean_path.exists():
            items.append(
                (
                    str(clean_path.resolve()),
                    str(noisy_path.resolve()),
                    "pre-paired",
                )
            )
    print(f"[OK] Găsite perechi VoiceBank: {len(items)}")
    return items


def process_librispeech(clean_dirs: Iterable[Path]) -> List[Tuple[str, str, str]]:
    """
    Generează intrări pentru augmentare din LibriSpeech:
      (clean_path, "N/A", "augment")
    """
    items: List[Tuple[str, str, str]] = []
    clean_dirs = list(clean_dirs)
    if not clean_dirs:
        return items

    names = [d.name for d in clean_dirs]
    print(f"[Info] Procesare LibriSpeech: {names}")

    for clean_dir in clean_dirs:
        _require_dir(clean_dir, f"LibriSpeech {clean_dir.name}")
        clean_files = find_files(clean_dir, extension=".flac")  # LibriSpeech e .flac
        if not clean_files:
            print(f"[W] Nu am găsit .flac în: {clean_dir}", file=sys.stderr)

        for clean_path in tqdm(clean_files, desc=f"  Scanare {clean_dir.name}", unit="file"):
            items.append(
                (
                    str(clean_path.resolve()),
                    "N/A",  # noisy va fi creat în etapa de încărcare/augmentare
                    "augment",
                )
            )
    print(f"[OK] Intrări LibriSpeech (augment): {len(items)}")
    return items


def main() -> None:
    # 0) Verifică directoare rădăcină
    for p, n in [
        (VOICEBANK_ROOT, "VOICEBANK_ROOT"),
        (LIBRISPEECH_ROOT, "LIBRISPEECH_ROOT"),
    ]:
        if not p.exists():
            print(f"[W] Directorul {n} nu există: {p}", file=sys.stderr)

    # 1) Asigură-te că folderul manifest există
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Creează manifestul de TRAINING
    print("[Step] Se generează manifestul de training...")
    vb_train_items = process_voicebank(VB_CLEAN_TRAIN, VB_NOISY_TRAIN)
    ls_train_items = process_librispeech([LS_CLEAN_TRAIN_100, LS_CLEAN_TRAIN_360])

    all_train_items = vb_train_items + ls_train_items
    if not all_train_items:
        print("[E] Niciun item de training detectat. Verifică căile!", file=sys.stderr)
        sys.exit(2)

    train_df = pd.DataFrame(
        all_train_items, columns=["clean_path", "noisy_path", "type"]
    )

    # Shuffle determinist pentru reproducibilitate
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    print(f"[OK] Am salvat {len(train_df)} iteme de training în: {TRAIN_CSV_PATH}")

    # 3) Creează manifestul de VALIDARE (doar VoiceBank pre-paired)
    print("\n[Step] Se generează manifestul de validare...")
    vb_val_items = process_voicebank(VB_CLEAN_VAL, VB_NOISY_VAL)

    if not vb_val_items:
        print("[E] Niciun item de validare VoiceBank detectat. Verifică căile!", file=sys.stderr)
        sys.exit(3)

    val_df = pd.DataFrame(vb_val_items, columns=["clean_path", "noisy_path", "type"])
    val_df.to_csv(VAL_CSV_PATH, index=False)
    print(f"[OK] Am salvat {len(val_df)} iteme de validare în: {VAL_CSV_PATH}")


if __name__ == "__main__":
    main()
