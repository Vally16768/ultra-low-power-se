#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import tensorflow as tf
from tensorflow.keras import layers as L, models
from scikeras.wrappers import KerasRegressor  # sklearn API

# --- add repo root to sys.path so "augment" is importable ---
import sys
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- import registrul de augmentări ---
import importlib
for m in [
    "augment.base",
    "augment.colored",
    "augment.babble",
    "augment.hum",
    "augment.bandlimited",
    "augment.bursts",
    "augment.reverb",
    "augment.channel",
]:
    importlib.import_module(m)
from augment.base import apply_chain

# ==================== Utils STFT ====================
def stft_mag(x, fs=16000, n_fft=512, hop=128):
    f, t, Z = stft(x, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    return Z, np.abs(Z)

def ideal_ratio_mask(m_noisy, m_clean, eps=1e-8):
    return np.clip((m_clean**2) / (m_noisy**2 + eps), 0.0, 1.0)

def inverse_from_mask(x_noisy, mask, fs=16000, n_fft=512, hop=128):
    _, _, Z = stft(x_noisy, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    Z_est = Z * mask
    _, y = istft(Z_est, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return y.astype(np.float32)

# ==================== Data sequences (on-the-fly) ====================
class TrainSequence(tf.keras.utils.Sequence):
    """Generează (X, y) pe loturi din WAV-uri curate + augmentări on-the-fly."""
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

        # pre-încarcă wav-urile (poți schimba pe "read-on-demand" dacă memoria e critică)
        self.cache = {}
        for p in self.paths_clean:
            x, fs = sf.read(p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(-1)
            if fs != sr:
                raise RuntimeError(f"Resample offline: {p} (fs={fs})")
            self.cache[p] = x

        # dimensiuni: F bins = n_fft//2+1; feature_dim = F*(2*ctx+1)
        self.F = self.n_fft // 2 + 1
        self.feat_dim = self.F * (2 * self.ctx + 1)

    def __len__(self):
        # numărul de batch-uri per epocă (poți crește pentru mai multe iterații)
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
        Xb = []
        Yb = []
        frames_needed = self.batch_frames
        while frames_needed > 0:
            p = self.rng.choice(self.paths_clean)
            clean = self._rand_seg(self.cache[p])

            # injectăm pool-ul pentru "babble" dacă apare în chain
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
    """Val pe VoiceBank DEMAND (perechi fixate) — fără augmentări."""
    def __init__(self, pairs, sr=16000, n_fft=512, hop=128, ctx=2, batch_frames=2048):
        self.pairs = [(Path(n), Path(c)) for n, c in pairs]
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.ctx = ctx
        self.batch_frames = batch_frames
        self.cache = []
        for pn, pc in self.pairs:
            xn, fs = sf.read(pn, dtype="float32", always_2d=False)
            xc, fs2 = sf.read(pc, dtype="float32", always_2d=False)
            if xn.ndim > 1:
                xn = xn.mean(-1)
            if xc.ndim > 1:
                xc = xc.mean(-1)
            if fs != self.sr or fs2 != self.sr:
                raise RuntimeError("Resample offline for DEMAND")
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

# ==================== Keras model (wrapped by sklearn) ====================
def build_keras(input_dim: int, output_dim: int, lr=1e-3, dropout=0.2):
    inp = L.Input(shape=(input_dim,), name="feat")
    x = L.LayerNormalization()(inp)
    x = L.Dense(512, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(output_dim, activation="sigmoid", name="mask")(x)  # 0..1
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return m

# ==================== Main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)   # manifests/train.csv
    ap.add_argument("--val", required=True)     # manifests/val_voicebank.csv
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--val_steps", type=int, default=20)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--segment_seconds", type=float, default=2.0)
    ap.add_argument("--batch_frames", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    # citește listele
    paths_clean = []
    with open(args.train) as f:
        for r in csv.DictReader(f):
            paths_clean.append(r["path_clean"])

    pairs = []
    with open(args.val) as f:
        for r in csv.DictReader(f):
            pairs.append((r["path_noisy"], r["path_clean"]))

    # secvențe
    tr_seq = TrainSequence(
        paths_clean,
        chain_config=None,
        sr=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        ctx=args.ctx,
        segment_seconds=args.segment_seconds,
        batch_frames=args.batch_frames,
        rng_seed=42,
    )
    va_seq = ValSequence(
        pairs,
        sr=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        ctx=args.ctx,
        batch_frames=args.batch_frames,
    )

    # dimensiuni
    input_dim = tr_seq.feat_dim
    output_dim = tr_seq.F

    # KerasRegressor (sklearn API) — IMPORTANT: y_required=False când X e Sequence care dă (X,y)
    reg = KerasRegressor(
        model=build_keras,
        model__input_dim=input_dim,
        model__output_dim=output_dim,
        model__lr=args.lr,
        model__dropout=args.dropout,
        epochs=args.epochs,
        verbose=1,
        batch_size=None,     # lucrăm pe cadre; batching-ul e în Sequence
        y_required=False,    # <--- cheie: nu cerem y separat; vine din Sequence
    )

    # antrenare
    reg.fit(
        tr_seq,
        validation_data=va_seq,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ],
    )

    # mică verificare obiectivă pe VoiceBank: SNR improvement (subset)
    def snr(a, b):
        n = a - b
        return 10 * np.log10((b**2).mean() / ((n**2).mean() + 1e-12))

    imps = []
    for (pn, pc) in pairs[:10]:  # primele 10 fișiere
        xn, fs = sf.read(pn, dtype="float32", always_2d=False)
        xc, _ = sf.read(pc, dtype="float32", always_2d=False)
        if xn.ndim > 1:
            xn = xn.mean(-1)
        if xc.ndim > 1:
            xc = xc.mean(-1)
        # extrage features pe tot semnalul zgomotos
        _, Mn = stft_mag(xn, fs=args.sr, n_fft=args.n_fft, hop=args.hop)
        Llog = np.log1p(Mn)
        # cadre cu context
        T, F = Llog.shape[1], Llog.shape[0]
        X = np.empty((T, tr_seq.feat_dim), np.float32)
        for t in range(T):
            stk = []
            for k in range(-args.ctx, args.ctx + 1):
                i = np.clip(t + k, 0, T - 1)
                stk.append(Llog[:, i])
            X[t] = np.concatenate(stk)
        # prezicere mască
        irm_hat = np.clip(reg.predict(X), 0, 1)  # [T,F]
        irm_hat = irm_hat.T  # [F,T]
        y_hat = inverse_from_mask(xn, irm_hat, fs=args.sr, n_fft=args.n_fft, hop=args.hop)[: len(xc)]
        imps.append(snr(y_hat, xc) - snr(xn, xc))
    print(f"Avg SNR improvement on VoiceBank (subset): {np.mean(imps):.2f} dB")

if __name__ == "__main__":
    main()
