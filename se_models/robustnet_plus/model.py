import torch
import torch.nn as nn
import torch.nn.functional as F

class ChLayerNorm(nn.Module):
    def __init__(self, ch): super().__init__(); self.ln = nn.GroupNorm(1, ch, eps=1e-5, affine=True)
    def forward(self, x): return self.ln(x)

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(ch, ch//r, 1)
        self.fc2 = nn.Conv1d(ch//r, ch, 1)
    def forward(self, x):
        s = self.avg(x)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class GatedDSConv(nn.Module):
    def __init__(self, ch, kernel, dilation, causal=False):
        super().__init__()
        pad = (kernel-1)*dilation
        if causal:
            self.pad = nn.ConstantPad1d((pad,0), 0.0)
        else:
            self.pad = nn.ConstantPad1d((pad//2, pad - pad//2), 0.0)
        self.dconv = nn.Conv1d(ch, ch, kernel, groups=ch, dilation=dilation, bias=False)
        self.pw_f = nn.Conv1d(ch, ch, 1, bias=False)
        self.pw_g = nn.Conv1d(ch, ch, 1, bias=False)
        self.norm = ChLayerNorm(ch)
        self.se = SEBlock(ch)

    def forward(self, x):
        y = self.pad(x)
        y = self.dconv(y)
        f = torch.tanh(self.pw_f(y))
        g = torch.sigmoid(self.pw_g(y))
        y = f * g
        y = self.norm(self.se(y))
        return y

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, causal=False):
        super().__init__()
        self.in1x1 = nn.Conv1d(ch, ch, 1)
        self.gds = GatedDSConv(ch, kernel, dilation, causal=causal)
        self.out1x1 = nn.Conv1d(ch, ch, 1)
        self.skip = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        res = x
        y = self.in1x1(x)
        y = self.gds(y)
        y = self.out1x1(y)
        return res + y, self.skip(y)

class RobustTCNPlusSE(nn.Module):
    def __init__(self, enc_dim=768, bottleneck=384, num_stages=3, blocks_per_stage=4, layers_per_block=8, kernel=17, causal=False):
        super().__init__()
        self.enc = nn.Conv1d(1, enc_dim, kernel_size=20, stride=10, padding=10)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck, 1)
        blocks = []
        for s in range(num_stages):
            for b in range(blocks_per_stage):
                for i in range(layers_per_block):
                    dilation = 2**i
                    blocks.append(TCNBlock(bottleneck, kernel, dilation=dilation, causal=causal))
        self.tcn = nn.ModuleList(blocks)
        self.mask = nn.Sequential(nn.Conv1d(bottleneck, enc_dim, 1), nn.Sigmoid())
        self.dec = nn.ConvTranspose1d(enc_dim, 1, kernel_size=20, stride=10, padding=10)

    def forward(self, x):  # x [B,1,T]
        z = self.enc(x)                # [B,E,T']
        h = self.bottleneck(z)         # [B,B,T']
        skip_sum = None
        for blk in self.tcn:
            h, s = blk(h)
            skip_sum = s if skip_sum is None else (skip_sum + s)
        m = self.mask(skip_sum)
        y = z * m
        out = self.dec(y)              # [B,1,T]
        if out.size(-1) > x.size(-1): out = out[..., :x.size(-1)]
        elif out.size(-1) < x.size(-1): out = F.pad(out, (0, x.size(-1) - out.size(-1)))
        return out
