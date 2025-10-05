import csv, random
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def _read_wav(path: str, sr: int) -> np.ndarray:
    x, r = sf.read(path, dtype="float32")
    if r != sr:
        raise RuntimeError(f"SR mismatch in {path}: {r} vs expected {sr}")
    if x.ndim == 2: x = x.mean(axis=1)
    return x

class Pairset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int, segment_seconds: Optional[float] = 2.0):
        self.items: List[Tuple[str,str]] = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["noisy"], row["clean"]))
        self.sr = sample_rate
        self.seg = segment_seconds

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        noisy_p, clean_p = self.items[idx]
        n = _read_wav(noisy_p, self.sr); c = _read_wav(clean_p, self.sr)
        L = min(len(n), len(c)); n = n[:L]; c = c[:L]
        if self.seg:
            seg_len = int(self.seg*self.sr)
            if L > seg_len:
                s = np.random.randint(0, L-seg_len+1); n=n[s:s+seg_len]; c=c[s:s+seg_len]
            elif L < seg_len:
                pad = seg_len - L; n=np.pad(n,(0,pad)); c=np.pad(c,(0,pad))
        return torch.from_numpy(n).float(), torch.from_numpy(c).float()

def collate(batch):
    xs, ys = zip(*batch)
    L = max(len(x) for x in xs)
    xs = [F.pad(x,(0, L-len(x))) for x in xs]
    ys = [F.pad(y,(0, L-len(y))) for y in ys]
    return torch.stack(xs,0), torch.stack(ys,0)
