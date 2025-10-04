import numpy as np
from pesq import pesq  # ITU-T P.862


# fs: 16000 -> WB, 8000 -> NB; alte fs trebuie resamplate înainte
def pesq_score(ref: np.ndarray, deg: np.ndarray, fs: int) -> float:
    mode = "wb" if fs >= 16000 else "nb"
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)
    # pesq() intern clipează [-1,1]; aliniază lungimile:
    n = min(len(ref), len(deg))
    return float(pesq(fs, ref[:n], deg[:n], mode))
