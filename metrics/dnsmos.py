from __future__ import annotations
from typing import Dict
import os

from metrics_utils import finite_or_default, clamp, LOG

try:
    # Official wrapper providing dnsmos_local.score_file()
    from dnsmos import dnsmos_local
except Exception as e:
    dnsmos_local = None
    _IMPORT_ERR = e

def dnsmos_wav(path: str) -> Dict[str, float]:
    """
    DNSMOS P.835 (no-reference). Expects a WAV readable by dnsmos_local.
    Returns: {'mos_sig': float, 'mos_bak': float, 'mos_ovr': float}
    """
    if dnsmos_local is None:
        raise ImportError(
            "The 'dnsmos' package (with dnsmos_local) is required. "
            "Install per the DNSMOS repo instructions."
        ) from _IMPORT_ERR

    if not isinstance(path, str) or not os.path.exists(path):
        raise FileNotFoundError(f"WAV not found: {path}")

    res = dnsmos_local.score_file(path)  # -> {'mos_sig','mos_bak','mos_ovr'}
    if not isinstance(res, dict) or not all(k in res for k in ("mos_sig", "mos_bak", "mos_ovr")):
        raise RuntimeError(f"Unexpected DNSMOS response: {res!r}")

    return {k: float(res[k]) for k in ("mos_sig", "mos_bak", "mos_ovr")}

def dnsmos_wav_safe(path: str, default: float = 2.5) -> Dict[str, float]:
    """
    Safe DNSMOS. If the backend fails or returns NaN/Inf, returns defaults.
    Outputs are clamped to [1.0, 5.0].
    """
    try:
        res = dnsmos_wav(path)
        mos_sig = clamp(finite_or_default(res["mos_sig"], default, "DNSMOS_SIG"), 1.0, 5.0)
        mos_bak = clamp(finite_or_default(res["mos_bak"], default, "DNSMOS_BAK"), 1.0, 5.0)
        mos_ovr = clamp(finite_or_default(res["mos_ovr"], default, "DNSMOS_OVR"), 1.0, 5.0)
        return {"mos_sig": mos_sig, "mos_bak": mos_bak, "mos_ovr": mos_ovr}
    except Exception as e:
        LOG.warning("DNSMOS failed for %s: %s. Using defaults.", path, e)
        return {"mos_sig": default, "mos_bak": default, "mos_ovr": default}
