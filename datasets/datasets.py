import torch
import torchaudio
import pandas as pd
import random
from torch.utils.data import Dataset
from pathlib import Path

def collate(batch):
    """Combină o listă de (noisy, clean) în două batch-uri."""
    noisy_wavs = [item[0] for item in batch]
    clean_wavs = [item[1] for item in batch]
    return torch.stack(noisy_wavs), torch.stack(clean_wavs)

def add_noise_snr(clean_wav, noise_wav, snr_db):
    """Adaugă zgomot la un semnal curat la un SNR dat (în dB)."""
    if clean_wav.shape[-1] < noise_wav.shape[-1]:
        # Zgomotul e mai lung, tăiem o bucată aleatorie
        start = random.randint(0, noise_wav.shape[-1] - clean_wav.shape[-1])
        noise_wav = noise_wav[..., start : start + clean_wav.shape[-1]]
    else:
        # Zgomotul e mai scurt, îl repetăm (pad)
        noise_wav = noise_wav.repeat(1, 1 + clean_wav.shape[-1] // noise_wav.shape[-1])
        noise_wav = noise_wav[..., : clean_wav.shape[-1]]

    # Calcul putere semnal și zgomot
    power_clean = torch.mean(clean_wav**2)
    power_noise = torch.mean(noise_wav**2)
    
    # Calculează factorul de scalare pentru zgomot
    snr_linear = 10**(snr_db / 10.0)
    scale = torch.sqrt(power_clean / (snr_linear * power_noise + 1e-8))

    # Combină semnalele
    noisy_wav = clean_wav + noise_wav * scale
    
    # Normalizează pentru a evita clipping-ul
    max_val = torch.max(torch.abs(noisy_wav))
    if max_val > 1.0:
        noisy_wav = noisy_wav / max_val
        
    return noisy_wav

class AugmentingPairset(Dataset):
    def __init__(
        self,
        csv_path,
        noise_dir, # Directorul cu zgomote pentru augmentare
        sample_rate=16000,
        segment_seconds=2.0,
        min_snr_db=-5.0, # SNR minim pentru augmentare
        max_snr_db=20.0  # SNR maxim pentru augmentare
    ):
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        
        # Încarcă manifestul principal
        self.df = pd.read_csv(csv_path)
        
        # Scanează și încarcă lista de fișiere de zgomot
        print(f"Se scanează folderul de zgomote: {noise_dir}")
        self.noise_files = list(Path(noise_dir).rglob("*.wav"))
        if not self.noise_files:
            print(f"Atenție: Nu am găsit fișiere .wav în {noise_dir}")
            
        print(f"Am găsit {len(self.noise_files)} fișiere de zgomot.")

    def __len__(self):
        return len(self.df)
        
    def _load_wav(self, path):
        """Încarcă și resamplează un fișier audio."""
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1: # Convert la mono
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav

    def _pad_or_crop(self, wav):
        """Aduce semnalul la lungimea fixă a segmentului."""
        L = wav.shape[-1]
        if L > self.segment_len:
            start = torch.randint(0, L - self.segment_len, (1,)).item()
            wav = wav[..., start : start + self.segment_len]
        elif L < self.segment_len:
            wav = torch.nn.functional.pad(wav, (0, self.segment_len - L))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        clean_path = row["clean_path"]
        clean_wav = self._load_wav(clean_path)

        if row["type"] == "pre-paired":
            # 1. Cazul "pre-paired" (Voicebank)
            noisy_path = row["noisy_path"]
            noisy_wav = self._load_wav(noisy_path)
            
        elif row["type"] == "augment":
            # 2. Cazul "augment" (Librispeech)
            # Alege un zgomot aleatoriu
            noise_path = random.choice(self.noise_files)
            noise_wav = self._load_wav(noise_path)
            
            # Alege un SNR aleatoriu
            snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
            
            # Adaugă zgomotul
            noisy_wav = add_noise_snr(clean_wav, noise_wav, snr_db)

        # Aplică padding/cropping la ambele
        clean_wav = self._pad_or_crop(clean_wav)
        noisy_wav = self._pad_or_crop(noisy_wav)

        return noisy_wav, clean_wav