import os, random, numpy as np, soundfile as sf
from pathlib import Path

SR = 16000
OUT = Path("data/noise/synth")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1337)

def white(n): 
    return rng.standard_normal(n).astype(np.float32) * 0.2

def brown(n):
    w = rng.standard_normal(n).astype(np.float32)
    return np.cumsum(w) / (np.arange(n, dtype=np.float32)+1).clip(1) * 0.05

def pink(n):
    X = rng.standard_normal(n).astype(np.float32)
    F = np.fft.rfft(X)
    freqs = np.fft.rfftfreq(n, 1/SR)
    F[1:] /= np.sqrt(freqs[1:])
    x = np.fft.irfft(F, n).astype(np.float32)
    x /= (np.max(np.abs(x)) + 1e-9)
    return x * 0.2

def save(x, path):
    sf.write(path, x, SR)

# 10 fișiere per tip, 60s fiecare
dur = 60
N = SR * dur
for i in range(10):
    save(white(N), OUT/f"white_{i:02d}.wav")
    save(pink(N),  OUT/f"pink_{i:02d}.wav")
    save(brown(N), OUT/f"brown_{i:02d}.wav")

# Babble: mix de fragmente random din Libri, dacă există
libri = []
for root, _, files in os.walk("data/clean_wav/libri"):
    for f in files:
        if f.lower().endswith(".wav") and ("dev-clean" in root or "train-clean" in root or "dev-other" in root):
            libri.append(os.path.join(root, f))
libri = sorted(libri)

if libri:
    import resampy
    def load_mono16k(p):
        x, s = sf.read(p, dtype="float32")
        if x.ndim > 1: x = x.mean(-1)
        if s != SR: x = resampy.resample(x, s, SR)
        return x

    for i in range(10):
        K = random.randint(5, 8)
        parts = []
        for _ in range(K):
            p = random.choice(libri)
            x = load_mono16k(p)
            if len(x) > N:
                st = random.randint(0, len(x) - N)
                x = x[st:st+N]
            elif len(x) < N:
                x = np.pad(x, (0, N-len(x)))
            parts.append(x)
        b = np.sum(parts, axis=0)
        b /= (np.max(np.abs(b)) + 1e-9)
        save(b*0.2, OUT/f"babble_{i:02d}.wav")

print("Done, noises at", OUT)
