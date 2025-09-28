import torch
import torch.nn as nn

class TinyMambaUNetStub(nn.Module):
    """
    Stub minim: conv1d encoder-decoder cu skip; NU e Mamba real.
    Doar ca smoke test pentru pipeline.
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(in_ch, base, 9, padding=4), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 9, stride=2, padding=4), nn.ReLU())
        self.bott = nn.Sequential(nn.Conv1d(base*2, base*2, 9, padding=4), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(base*2, base, 4, stride=2, padding=1), nn.ReLU())
        self.out  = nn.Conv1d(base, in_ch, 1)

    def forward(self, x):
        # x: (B, T) sau (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b  = self.bott(e2)
        d1 = self.dec1(b)
        diff = e1.size(-1) - d1.size(-1)
        if diff > 0:
            d1 = nn.functional.pad(d1, (0, diff))
        elif diff < 0:
            d1 = d1[..., :e1.size(-1)]
        y  = self.out(d1 + e1)
        return y.squeeze(1)

def build_model(cfg: dict):
    mcfg = cfg.get("model", {})
    in_ch = int(mcfg.get("in_ch", 1))
    base  = int(mcfg.get("base", 32))
    return TinyMambaUNetStub(in_ch=in_ch, base=base)
