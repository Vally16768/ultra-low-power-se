# DNSMOS P.835 – non-intruziv (no-reference)
# Input: path la wav (16 kHz mono); Output: dict cu MOS sig/noise/overall
import numpy as np, soundfile as sf, onnxruntime as ort
from dnsmos import dnsmos_local  # folosește modelul pre-ambalat


def dnsmos_wav(path: str) -> dict:
    res = dnsmos_local.score_file(path)  # {"mos_sig":..., "mos_bak":..., "mos_ovr":...}
    return {k: float(v) for k, v in res.items()}
