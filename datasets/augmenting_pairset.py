# datasets/augmenting_pairset.py
# -*- coding: utf-8 -*-
"""
AugmentingPairset care:
  - pentru rânduri type=="pre-paired" (ex. VoiceBank): citește perechea noisy/clean ca atare
  - pentru rânduri type=="augment" (ex. LibriSpeech): produce noisy din clean fie:
        (A) prin mix cu zgomote "pure" (mix_noise), fie
        (B) printr-un lanț de efecte definit în augment/* (chain via Registry)
  - are RNG determinist, crop/pad la segment fix, și collate simplu [B,1,T]

Necesită:
  - torchaudio, torch
  - folderul augment/ cu modulele: base.py, babble.py, colored.py, bandlimited.py,
    bursts.py, channel.py, hum.py, reverb.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# === Importăm registrul și "înregistrăm" transform-urile prin import side-effects ===
from augment.base import apply_chain, Registry  # noqa: F401

# IMPORTANT: aceste importuri populază Registry prin decoratorul @register
import augment.babble as _aug_babble  # noqa: F401
import augment.bandlimited as _aug_bandlimited  # noqa: F401
import augment.bursts as _aug_bursts  # noqa: F401
import augment.channel as _aug_channel  # noqa: F401
import augment.colored as _aug_colored  # noqa: F401
import augment.hum as _aug_hum  # noqa: F401
import augment.reverb as _aug_reverb  # noqa: F401


# ------------------------------
# Utils: collate și mix la SNR
# ------------------------------
def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combină o listă de (noisy, clean) în două batch-uri [B,1,T]."""
    noisy_wavs = [b[0] for b in batch]
    clean_wavs = [b[1] for b in batch]
    return torch.stack(noisy_wavs), torch.stack(clean_wavs)


def add_noise_snr(clean_wav: torch.Tensor, noise_wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Adaugă zgomot la un semnal curat la un SNR dat (în dB).
    Intrări [1, T]. Returnează [1, T].
    """
    # Ajustează lungimea zgomotului
    if clean_wav.shape[-1] < noise_wav.shape[-1]:
        start = random.randint(0, noise_wav.shape[-1] - clean_wav.shape[-1])
        noise_wav = noise_wav[..., start : start + clean_wav.shape[-1]]
    else:
        reps = 1 + clean_wav.shape[-1] // noise_wav.shape[-1]
        noise_wav = noise_wav.repeat(1, reps)[..., : clean_wav.shape[-1]]

    power_clean = torch.mean(clean_wav**2)
    power_noise = torch.mean(noise_wav**2) + 1e-12

    snr_linear = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(power_clean / (snr_linear * power_noise))
    noisy_wav = clean_wav + noise_wav * scale

    # Normalizează pentru a evita clipping-ul
    max_val = torch.max(torch.abs(noisy_wav))
    if max_val > 1.0:
        noisy_wav = noisy_wav / max_val
    return noisy_wav


# ------------------------------
# Dataset cu două moduri de augment
# ------------------------------
class AugmentingPairset(Dataset):
    """
    Dataset:
      - rânduri "pre-paired": citește noisy/clean
      - rânduri "augment": produce noisy din clean
           mod "mix_noise": clean + zgomot din noise_dir la SNR random
           mod "chain":     aplică un lanț de efecte din augment/* (Registry)

    Parametri cheie:
      csv_path: manifest cu coloane ["clean_path","noisy_path","type"]
      noise_dir: director cu WAV-uri de zgomot "pure" (pt mix_noise)
      aug_mode: "mix_noise" | "chain" | "auto"
          - "mix_noise": folosește strict amestec cu zgomote
          - "chain":     folosește strict lanțul din aug_chain
          - "auto":      dacă aug_chain e setat -> chain, altfel mix_noise
      aug_chain: listă de pași pentru Registry.apply_chain (vezi exemplu mai jos)
          ex:
            [
              {"name": "reverb_toy"},
              {"name": "add_colored_noise", "params": {"color":"pink", "snr_db": 5}},
              {"name": "eq_tilt", "params": {"db_per_oct": 3}},
            ]
      babble_pool_dir: dacă în lanț folosești "add_babble", atunci params trebuie să includă
          "pool_paths": list[str]. Dacă nu vrei să pui lista în manifest, poți seta babble_pool_dir
          iar dataset o va popula automat în __init__.

    Notă: toate transform-urile din augment/* sunt pure-numpy, deci convertim
          torch.Tensor ↔ numpy.ndarray pe durata aplicării lanțului.
    """

    def __init__(
        self,
        csv_path: str,
        noise_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        min_snr_db: float = -5.0,
        max_snr_db: float = 20.0,
        seed: int = 123,
        aug_mode: str = "auto",  # "mix_noise" | "chain" | "auto"
        aug_chain: Optional[List[Dict[str, Any]]] = None,
        babble_pool_dir: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.aug_mode = aug_mode
        self.aug_chain = aug_chain or []

        # manifest
        self.df = pd.read_csv(csv_path)

        # zgomote pentru mix_noise
        self.noise_files: List[Path] = []
        if noise_dir:
            self.noise_files = list(Path(noise_dir).rglob("*.wav"))
            if not self.noise_files:
                print(f"[WARN] Nu am găsit zgomote WAV în noise_dir: {noise_dir}")

        # pool pentru babble (dacă e nevoie)
        self.babble_pool_paths: List[str] = []
        if babble_pool_dir:
            self.babble_pool_paths = [str(p) for p in Path(babble_pool_dir).rglob("*.wav")]
            if not self.babble_pool_paths:
                print(f"[WARN] Nu am găsit fișiere WAV în babble_pool_dir: {babble_pool_dir}")

        # RNG determinist
        self._seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        # numpy RNG pentru chain
        import numpy as _np
        self._rng = _np.random.default_rng(seed)

        # sanity: dacă ai ales "chain" dar nu ai definit aug_chain → warning
        if self.aug_mode == "chain" and not self.aug_chain:
            print("[WARN] aug_mode='chain' dar aug_chain este gol. Nu se vor aplica efecte.")

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------
    # Loader audio + pregătire
    # ------------------------------
    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:  # stereo -> mono
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav

    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        L = wav.shape[-1]
        if L > self.segment_len:
            # limitele pentru randint trebuie să fie valide (include capătul)
            max_start = max(0, L - self.segment_len)
            start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
            wav = wav[..., start : start + self.segment_len]
        elif L < self.segment_len:
            wav = torch.nn.functional.pad(wav, (0, self.segment_len - L))
        return wav

    # ------------------------------
    # Augmentări
    # ------------------------------
    def _apply_chain_numpy(self, clean_wav: torch.Tensor) -> torch.Tensor:
        """
        Aplică lanțul din augment/* pe semnalul clean și returnează 'noisy' (torch).
        """
        sr = self.sample_rate
        x = clean_wav.squeeze(0).detach().cpu().numpy().astype("float32")  # [T]
        chain = self._prepare_chain_params(self.aug_chain)
        y = apply_chain(x, sr=sr, chain=chain, rng=self._rng)  # numpy → numpy
        y_t = torch.from_numpy(y).view(1, -1)  # [1,T]
        return y_t

    def _prepare_chain_params(self, chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Completează parametri lipsă (ex. pool_paths pentru add_babble) dacă avem babble_pool_dir.
        """
        if not chain:
            return chain
        out = []
        for step in chain:
            st = dict(step)
            name = st.get("name")
            params = dict(st.get("params", {}))
            if name == "add_babble" and "pool_paths" not in params and self.babble_pool_paths:
                # adaugă implicit un pool dacă nu s-a trecut deja
                params["pool_paths"] = self.babble_pool_paths
            st["params"] = params
            out.append(st)
        return out

    def _make_noisy_from_mix(self, clean_wav: torch.Tensor) -> torch.Tensor:
        """
        Produce noisy = clean + noise @ SNR random folosind fișiere din noise_dir.
        Fallback: dacă nu există zgomote, întoarce clean (identitate).
        """
        if not self.noise_files:
            return clean_wav.clone()
        noise_path = str(random.choice(self.noise_files))
        noise_wav = self._load_wav(noise_path)
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        return add_noise_snr(clean_wav, noise_wav, snr_db)

    # ------------------------------
    # Get item
    # ------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        typ = str(row["type"])
        clean_path = str(row["clean_path"])
        clean_wav = self._load_wav(clean_path)

        if typ == "pre-paired":
            noisy_path = str(row["noisy_path"])
            noisy_wav = self._load_wav(noisy_path)

        else:
            # "augment": decide între chain / mix_noise (sau auto)
            mode = self.aug_mode
            if mode == "auto":
                mode = "chain" if self.aug_chain else "mix_noise"

            if mode == "chain":
                noisy_wav = self._apply_chain_numpy(clean_wav)
            elif mode == "mix_noise":
                noisy_wav = self._make_noisy_from_mix(clean_wav)
            else:
                raise ValueError(f"aug_mode necunoscut: {self.aug_mode!r}")

        # Crop/pad
        clean_wav = self._pad_or_crop(clean_wav)
        noisy_wav = self._pad_or_crop(noisy_wav)

        return noisy_wav, clean_wav
