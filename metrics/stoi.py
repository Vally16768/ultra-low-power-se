import numpy as np
from pystoi import stoi


def stoi_score(ref: np.ndarray, deg: np.ndarray, fs: int) -> float:
    n = min(len(ref), len(deg))
    return float(stoi(ref[:n].astype(np.float32), deg[:n].astype(np.float32), fs, extended=False))
