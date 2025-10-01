import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class TinyMambaUNetStub(nn.Module):
    """
    Stub minim pentru smoke-test: conv1d encoder/decoder cu skip.
    NU implementează Mamba real; scopul e să verifice pipeline-ul cap-coadă.
    """
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(base, base * 2, kernel_size=9, stride=2, padding=4),
            nn.ReLU()
        )
        self.bott = nn.Sequential(
            nn.Conv1d(base * 2, base * 2, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.out = nn.Conv1d(base, in_ch, kernel_size=1)

    @staticmethod
    def _align_time(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Aliniază x la lungimea temporală a lui ref pe ultima dimensiune,
        folosind control-flow pe int (TorchScript/ONNX safe).
        """
        T_ref: int = int(ref.size(-1))
        T_x: int = int(x.size(-1))

        if T_x > T_ref:                      # crop
            x = x[..., :T_ref]
        pad_amt: int = T_ref - T_x           # pad dacă e nevoie
        if pad_amt > 0:
            x = F.pad(x, (0, pad_amt))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Acceptă (B, T) sau (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (B, 1, T)

        e1 = self.enc1(x)      # (B, base, T1)
        e2 = self.enc2(e1)     # (B, 2*base, ~T1/2)
        b  = self.bott(e2)     # (B, 2*base, ~T1/2)
        d1 = self.dec1(b)      # (B, base, ~T1)
        d1 = self._align_time(e1, d1)

        y  = self.out(d1 + e1)     # (B, 1, T1)
        return y.squeeze(1)        # (B, T1) — pentru training/infer Torch

def build_model(cfg: Dict) -> nn.Module:
    mcfg = cfg.get("model", {}) or {}
    in_ch = int(mcfg.get("in_ch", 1))
    base  = int(mcfg.get("base", 32))
    return TinyMambaUNetStub(in_ch=in_ch, base=base)
