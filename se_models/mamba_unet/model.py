import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


def _same(x: torch.Tensor, y_len: int) -> torch.Tensor:
    diff = int(y_len) - int(x.size(-1))
    if diff == 0:
        return x
    if diff > 0:
        return F.pad(x, (0, diff))
    return x[..., :y_len]


class ConvGNAct(nn.Module):
    def __init__(self, c_in, c_out, k=5, s=1, causal=False):
        super().__init__()
        pad_l = k - 1 if causal else (k - 1) // 2
        pad_r = 0 if causal else (k - 1) - (k - 1) // 2
        self.pad = (pad_l, pad_r)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=0, bias=True)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=c_out)  # stabil la B mic
        self.act = nn.PReLU(c_out)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = F.pad(x, self.pad)
        x = self.conv(x)
        x = self.gn(x)
        return self.act(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out, causal=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(c_in, c_out, k=5, s=2, causal=causal),  # downsample x2
            ConvGNAct(c_out, c_out, k=5, s=1, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, c_in, c_out, causal=False):
        super().__init__()
        self.up = nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=True)
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="leaky_relu")
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        self.conv = nn.Sequential(
            ConvGNAct(c_out * 2, c_out, k=5, s=1, causal=causal),
            ConvGNAct(c_out, c_out, k=5, s=1, causal=causal),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = _same(x, int(skip.size(-1)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SmallUNet1D(nn.Module):
    """[B,C_in,T] -> [B,1,T]; opțional cauzal; ONNX-friendly."""

    def __init__(self, in_ch=1, base=32, depth=3, causal: bool = False):
        super().__init__()
        self.causal = causal
        c = base
        self.enc1 = nn.Sequential(ConvGNAct(in_ch, c, causal=causal), ConvGNAct(c, c, causal=causal))
        self.downs = nn.ModuleList()
        ch = [c]
        for _ in range(depth):
            self.downs.append(Down(c, c * 2, causal=causal))
            c *= 2
            ch.append(c)
        self.bottleneck = nn.Sequential(ConvGNAct(c, c, causal=causal), ConvGNAct(c, c, causal=causal))
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(c, c // 2, causal=causal))
            c //= 2
        self.head = nn.Conv1d(c, 1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        T = int(x.size(-1))
        s1 = self.enc1(x)
        skips = [s1]
        h = s1
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = self.bottleneck(h)
        for i, u in enumerate(self.ups, 1):
            h = u(h, skips[-(i + 1)])
        y = self.head(h)
        return _same(y, T)


def build_model(cfg: Dict[str, Any] | None = None) -> nn.Module:
    m = (cfg or {}).get("model", {}) or {}
    base = int(m.get("base_channels", m.get("base", 32)))
    depth = int(m.get("depth", 3))
    causal = bool(m.get("causal", False))
    in_ch = int(m.get("in_ch", 1))
    return SmallUNet1D(in_ch=in_ch, base=base, depth=depth, causal=causal)
