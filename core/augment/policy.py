# core/augment/policy.py
from __future__ import annotations
from typing import List, Dict, Any, Callable
import inspect
import numpy as np

from augment.base import Registry
import augment.babble as _aug_babble      # noqa: F401
import augment.bandlimited as _aug_band   # noqa: F401
import augment.bursts as _aug_bursts      # noqa: F401
import augment.channel as _aug_channel    # noqa: F401
import augment.colored as _aug_colored    # noqa: F401
import augment.hum as _aug_hum            # noqa: F401
import augment.reverb as _aug_reverb      # noqa: F401

def _filter_kwargs(fn: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys()) - {"x", "sr", "rng"}
        return {k: v for k, v in (params or {}).items() if k in allowed}
    except Exception:
        return {}

def _sanitize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32, order="C")
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(y, -1.0, 1.0)

def apply_chain(clean: np.ndarray, sr: int, chain: List[Dict[str, Any]], rng) -> np.ndarray:
    y = _sanitize(clean)
    for step in (chain or []):
        name = step.get("name")
        if not name:
            continue
        fn = Registry.get(name)
        if fn is None:
            continue
        params = _filter_kwargs(fn, step.get("params", {}) or {})
        try:
            y = fn(y, sr=sr, rng=rng, **params)
        except TypeError:
            y = fn(y, sr=sr, rng=rng)
        y = _sanitize(y)
    return y
