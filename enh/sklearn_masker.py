# enh/sklearn_masker.py
import csv, soundfile as sf, numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from .utils import stft_mag, ideal_ratio_mask, inverse_from_mask
from augment.base import apply_chain
import importlib  # ensure augment modules are registered

# import all augment submodules (simple eager import)
for m in ["augment.colored", "augment.babble", "augment.hum", "augment.bandlimited", "augment.bursts", "augment.reverb", "augment.channel"]:
    importlib.import_module(m)


class MaskerRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        sr=16000,
        n_fft=512,
        hop=128,
        ctx=2,
        segment_seconds=2.0,
        samples_per_file=2,
        chain_config: List[Dict] = None,
        base_estimator=None,
        babble_pool: List[str] | None = None,
        random_state=0,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.ctx = ctx
        self.segment_seconds = segment_seconds
        self.samples_per_file = samples_per_file
        self.chain_config = chain_config or [{"name": "reverb_toy"}, {"name": "add_colored_noise", "params": {"color": "pink", "snr_db": 5}}]
        self.base_estimator = base_estimator or Ridge(alpha=1.0)
        self.model = MultiOutputRegressor(self.base_estimator)
        self.babble_pool = babble_pool
        self.random_state = random_state

    def _seg_random(self, x):
        L = len(x)
        S = int(self.segment_seconds * self.sr)
        if L <= S:
            pad = np.zeros(S, dtype=np.float32)
            pad[:L] = x
            return pad
        s = np.random.default_rng(self.random_state).integers(0, L - S)
        return x[s : s + S].astype(np.float32)

    def _frames_ctx(self, L_log):
        T, F = L_log.shape[1], L_log.shape[0]
        frames = []
        for t in range(T):
            stack = []
            for k in range(-self.ctx, self.ctx + 1):
                i = np.clip(t + k, 0, T - 1)
                stack.append(L_log[:, i])
            frames.append(np.concatenate(stack))
        return np.stack(frames)  # [T, F*(2*ctx+1)]

    def fit(self, train_csv: str, babble_pool_csv: str | None = None):
        rng = np.random.default_rng(self.random_state)
        # pool pentru babble (opțional): listă de căi WAV/FLAC
        if babble_pool_csv:
            self.babble_pool = [row["path_clean"] for row in csv.DictReader(open(babble_pool_csv))]

        X_frames = []
        Y_frames = []
        with open(train_csv) as f:
            for row in csv.DictReader(f):
                path = Path(row["path_clean"])
                x, fs = sf.read(path, dtype="float32", always_2d=False)
                if x.ndim > 1:
                    x = x.mean(-1)
                if fs != self.sr:
                    raise RuntimeError("Resample offline pentru simplitate.")
                for _ in range(self.samples_per_file):
                    clean = self._seg_random(x)
                    # completează param pool pentru babble dacă e nevoie
                    chain = []
                    for step in self.chain_config:
                        s = dict(step)  # shallow copy
                        if s["name"] == "add_babble":
                            s.setdefault("params", {})["pool_paths"] = self.babble_pool
                        chain.append(s)
                    noisy = apply_chain(clean, self.sr, chain, rng)
                    Zc, Mc = stft_mag(clean, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
                    Zn, Mn = stft_mag(noisy, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
                    irm = ideal_ratio_mask(Mn, Mc)
                    L = np.log1p(Mn)
                    feats = self._frames_ctx(L)
                    X_frames.append(feats)
                    Y_frames.append(irm.T)  # [T,F]
        X = np.concatenate(X_frames, axis=0)
        Y = np.concatenate(Y_frames, axis=0)
        self.model.fit(X, Y)
        return self

    def predict_wave(self, noisy_wave: np.ndarray):
        Zn, Mn = stft_mag(noisy_wave, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
        L = np.log1p(Mn)
        X = self._frames_ctx(L)
        irm_hat = np.clip(self.model.predict(X), 0.0, 1.0)  # [T,F]
        irm_hat = irm_hat.T  # [F,T]
        y = inverse_from_mask(noisy_wave, irm_hat, fs=self.sr, n_fft=self.n_fft, hop=self.hop)
        return y
