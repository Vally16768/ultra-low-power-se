# deploy/streaming.py
from __future__ import annotations
import numpy as np


def ola_frames(x_len, win, hop):
    idx = []
    s = 0
    while s < x_len:
        e = s + win
        idx.append((s, e))
        if e >= x_len:
            break
        s += hop
    return idx


def overlap_add(frames, win, hop, out_len=None):
    if out_len is None:
        out_len = (len(frames) - 1) * hop + win
    y = np.zeros(out_len, dtype=frames[0].dtype)
    w = np.zeros(out_len, dtype=np.float32)
    for i, f in enumerate(frames):
        s = i * hop
        e = s + win
        y[s:e] += f
        w[s:e] += 1.0
    w[w == 0] = 1.0
    return y / w


def enhance_streaming_onnx(wav: np.ndarray, sess, win=20480, hop=10240):
    """
    wav: mono float32 in [-1,1]; sess: ORT session cu input 'noisy' [1,1,T]
    """
    assert wav.ndim == 1
    frames = []
    for s, e in ola_frames(len(wav), win, hop):
        chunk = wav[s:e]
        if len(chunk) < win:
            pad = np.zeros(win, dtype=wav.dtype)
            pad[: len(chunk)] = chunk
            chunk = pad
        x = chunk[None, None, :].astype(np.float32)
        y = sess.run(["enhanced"], {"noisy": x})[0][0, 0]  # [1,1,T] -> [T]
        frames.append(y[: len(chunk)])
    out = overlap_add(frames, win, hop, out_len=len(wav))
    return np.clip(out, -1.0, 1.0)
