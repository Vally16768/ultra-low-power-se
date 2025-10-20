# augment/base.py
from typing import Callable, Dict, Any
import numpy as np

Registry: Dict[str, Callable[..., np.ndarray]] = {}


def register(name: str):
    def deco(fn: Callable[..., np.ndarray]):
        Registry[name] = fn
        return fn

    return deco


def apply_chain(x: np.ndarray, sr: int, chain: list[dict], rng: np.random.Generator) -> np.ndarray:
    """
    chain = [
      {"name": "reverb_toy", "params": {...}},
      {"name": "add_colored_noise", "params": {"color":"pink","snr_db":5}},
      ...
    ]
    """
    y = x.copy()
    for step in chain:
        fn = Registry[step["name"]]
        y = fn(y, sr=sr, rng=rng, **step.get("params", {}))
    return np.clip(y, -1, 1)
