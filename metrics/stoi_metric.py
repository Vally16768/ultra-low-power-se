from pystoi.stoi import stoi as _stoi
import numpy as np


def stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    """
    Calculează STOI (sau eSTOI dacă extended=True).
    """
    return float(_stoi(ref.astype(np.float32), deg.astype(np.float32), sr, extended=extended))
