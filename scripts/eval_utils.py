from __future__ import annotations
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from .features import stft_mag, inverse_from_mask

def snr(a, b):
    n = a - b
    return 10 * np.log10((b**2).mean() / ((n**2).mean() + 1e-12))

def quick_vb_snr_improvement(pairs, reg, sr=16000, n_fft=512, hop=128, ctx=2, feat_dim=None):
    """Compute average SNR improvement on a subset with a progress bar."""
    imps = []
    for pn, pc in tqdm(pairs, desc="Eval SNR (subset)"):
        xn, fs = sf.read(pn, dtype="float32", always_2d=False)
        xc, _ = sf.read(pc, dtype="float32", always_2d=False)
        if xn.ndim > 1: xn = xn.mean(-1)
        if xc.ndim > 1: xc = xc.mean(-1)

        _, Mn = stft_mag(xn, fs=sr, n_fft=n_fft, hop=hop)
        Llog = np.log1p(Mn)
        T = Llog.shape[1]
        X = np.empty((T, feat_dim), np.float32)
        for t in range(T):
            stk = []
            for k in range(-ctx, ctx + 1):
                i = np.clip(t + k, 0, T - 1)
                stk.append(Llog[:, i])
            X[t] = np.concatenate(stk)
        irm_hat = np.clip(reg.predict(X), 0, 1)  # [T,F]
        irm_hat = irm_hat.T  # [F,T]
        y_hat = inverse_from_mask(xn, irm_hat, fs=sr, n_fft=n_fft, hop=hop)[: len(xc)]
        imps.append(snr(y_hat, xc) - snr(xn, xc))
    return float(np.mean(imps)) if imps else float("nan")
