import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_convergence(pred_mag, target_mag, eps: float = 1e-7):
    return torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + eps)

class STFT(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft; self.hop_length = hop_length; self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
    def forward(self, x: torch.Tensor):
        if x.dim()==3: x=x.squeeze(1)
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)

class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256,512,1024), hop_sizes=(64,128,256), win_lengths=(256,512,1024), mag_weight=0.5, sc_weight=0.5):
        super().__init__()
        self.stfts = nn.ModuleList([STFT(nf,hs,wl) for nf,hs,wl in zip(fft_sizes,hop_sizes,win_lengths)])
        self.mag_weight=mag_weight; self.sc_weight=sc_weight
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mag_loss = 0.0; sc_loss = 0.0
        for stft in self.stfts:
            P = stft(pred); T = stft(target)
            mag_loss += F.l1_loss(P.abs(), T.abs())
            sc_loss += spectral_convergence(P.abs(), T.abs())
        return self.sc_weight*sc_loss/len(self.stfts), self.mag_weight*mag_loss/len(self.stfts)

def sisdr_loss(est: torch.Tensor, ref: torch.Tensor, eps=1e-8):
    if est.dim()==3: est=est.squeeze(1)
    if ref.dim()==3: ref=ref.squeeze(1)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    s = torch.sum(ref*est, dim=-1, keepdim=True) * ref / (torch.sum(ref**2, dim=-1, keepdim=True) + eps)
    e = est - s
    sdr = 10*torch.log10((torch.sum(s**2, dim=-1) + eps)/(torch.sum(e**2, dim=-1) + eps))
    return -sdr.mean()

def dc_offset_loss(x: torch.Tensor):
    if x.dim()==3: x=x.squeeze(1)
    return x.mean(dim=-1).abs().mean()
