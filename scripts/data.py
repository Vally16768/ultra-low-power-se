from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import tensorflow as tf
from .features import stft_mag, ideal_ratio_mask

class TrainSequence(tf.keras.utils.Sequence):
    """Generates (X, y) batches from clean WAVs with on-the-fly augmentations."""
    def __init__(
        self,
        paths_clean,
        chain_config,
        sr=16000,
        n_fft=512,
        hop=128,
        ctx=2,
        segment_seconds=2.0,
        batch_frames=2048,
        rng_seed=42,
        babble_pool=None,
    ):
        self.paths_clean = [Path(p) for p in paths_clean]
        if not self.paths_clean:
            raise RuntimeError("Empty training file list.")
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.ctx = ctx
        self.S = int(segment_seconds * sr)
        self.batch_frames = batch_frames
        self.rng = np.random.default_rng(rng_seed)
        self.chain_config = chain_config or [
            {"name": "reverb_toy"},
            {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 5}},
            {"name": "bandlimit_noise", "params": {"snr_db": 8}},
            {"name": "add_hum", "params": {"snr_db": 12}},
        ]
        self.babble_pool = babble_pool

        # pre-load wavs with a progress bar
        self.cache = {}
        for p in tqdm(self.paths_clean, desc="Preloading clean WAVs", leave=False):
            x, fs = sf.read(p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(-1)
            if fs != sr:
                raise RuntimeError(f"Resample offline: {p} (fs={fs}, expected {sr})")
            self.cache[p] = x

        self.F = self.n_fft // 2 + 1
        self.feat_dim = self.F * (2 * self.ctx + 1)

    def __len__(self):
        return max(1, len(self.paths_clean))

    def _rand_seg(self, x):
        if len(x) <= self.S:
            y = np.zeros(self.S, np.float32)
            y[: len(x)] = x
            return y
        s = self.rng.integers(0, len(x) - self.S)
        return x[s : s + self.S].astype(np.float32)

    def _frames_ctx(self, Llog):
        T, F = Llog.shape[1], Llog.shape[0]
        feats = np.empty((T, self.feat_dim), dtype=np.float32)
        for t in range(T):
            stk = []
            for k in range(-self.ctx, self.ctx + 1):
                i = np.clip(t + k, 0, T - 1)
                stk.append(Llog[:, i])
            feats[t] = np.concatenate(stk)
        return feats

    def __getitem__(self, idx):
        # import augment registry late to keep import time low
        from augment.base import apply_chain

        Xb = []
        Yb = []
        frames_needed = self.batch_frames
        while frames_needed > 0:
            p = self.rng.choice(self.paths_clean)
            clean = self._rand_seg(self.cache[p])

            chain = []
            for step in self.chain_config:
                s = dict(step)
                if s["name"] == "add_babble":
                    s.setdefault("params", {})["pool_paths"] = self.babble_pool or [str(pp) for pp in self.paths_clean]
                chain.append(s)

            noisy = apply_chain(clean, self.sr, chain, self.rng)
            _, Mc = stft_mag(clean, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
            _, Mn = stft_mag(noisy, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
            irm = ideal_ratio_mask(Mn, Mc)  # [F,T]
            Llog = np.log1p(Mn)
            feats = self._frames_ctx(Llog)  # [T, feat_dim]
            Xb.append(feats.astype(np.float32))
            Yb.append(irm.T.astype(np.float32))  # [T, F]
            frames_needed -= feats.shape[0]
        X = np.concatenate(Xb, axis=0)[: self.batch_frames]
        Y = np.concatenate(Yb, axis=0)[: self.batch_frames]
        return X, Y

class ValSequence(tf.keras.utils.Sequence):
    """Fixed pairs (noisy, clean) — VoiceBank DEMAND-like."""
    def __init__(self, pairs, sr=16000, n_fft=512, hop=128, ctx=2, batch_frames=2048):
        self.pairs = [(Path(n), Path(c)) for n, c in pairs]
        if not self.pairs:
            raise RuntimeError("Empty validation pair list.")
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.ctx = ctx
        self.batch_frames = batch_frames
        self.cache = []
        for pn, pc in tqdm(self.pairs, desc="Preloading val pairs", leave=False):
            xn, fs = sf.read(pn, dtype="float32", always_2d=False)
            xc, fs2 = sf.read(pc, dtype="float32", always_2d=False)
            if xn.ndim > 1: xn = xn.mean(-1)
            if xc.ndim > 1: xc = xc.mean(-1)
            if fs != self.sr or fs2 != self.sr:
                raise RuntimeError(f"Resample offline for DEMAND: {pn} or {pc}")
            self.cache.append((xn, xc))
        self.F = self.n_fft // 2 + 1
        self.feat_dim = self.F * (2 * self.ctx + 1)

    def __len__(self):
        return max(1, len(self.cache))

    def _frames_ctx(self, Llog):
        T, F = Llog.shape[1], Llog.shape[0]
        feats = np.empty((T, self.feat_dim), np.float32)
        for t in range(T):
            stk = []
            for k in range(-self.ctx, self.ctx + 1):
                i = np.clip(t + k, 0, T - 1)
                stk.append(Llog[:, i])
            feats[t] = np.concatenate(stk)
        return feats

    def __getitem__(self, idx):
        Xb = []
        Yb = []
        frames_needed = self.batch_frames
        i = idx % len(self.cache)
        xn, xc = self.cache[i]
        _, Mc = stft_mag(xc, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
        _, Mn = stft_mag(xn, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
        irm = ideal_ratio_mask(Mn, Mc)
        Llog = np.log1p(Mn)
        feats = self._frames_ctx(Llog)
        while frames_needed > 0:
            Xb.append(feats.astype(np.float32))
            Yb.append(irm.T.astype(np.float32))
            frames_needed -= feats.shape[0]
        X = np.concatenate(Xb, axis=0)[: self.batch_frames]
        Y = np.concatenate(Yb, axis=0)[: self.batch_frames]
        return X, Y
