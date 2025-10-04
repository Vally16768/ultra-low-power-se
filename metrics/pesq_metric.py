from pesq import pesq as _pesq
import numpy as np


def pesq_score(ref: np.ndarray, deg: np.ndarray, sr: int = 16000, mode: str = "wb") -> float:
    """
    Calculează PESQ între referință (clean) și semnalul degradat/enhanced.
    - sr: 16000 recomandat
    - mode: "wb" (wideband, 16 kHz) sau "nb" (8 kHz)
    """
    return _pesq(sr, ref.astype(np.float32), deg.astype(np.float32), mode)
