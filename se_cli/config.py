from __future__ import annotations
import copy
import yaml
from typing import Any, Dict

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

def _auto_cast_scalar(val: str):
    s = val.strip()
    low = s.lower()
    if low in ("true","false"):
        return low == "true"
    try:
        if s.startswith("0x"):
            return int(s, 16)
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except Exception:
        return s

def kv_to_nested_dict(pairs: Dict[str, str] | None) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for full_k, v in (pairs or {}).items():
        cur = root
        keys = full_k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = _auto_cast_scalar(v) if isinstance(v, str) else v
    return root

def load_config(config_path: str, override_kv: Dict[str, str] | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    overrides = kv_to_nested_dict(override_kv or {})
    return _merge_dicts(cfg, overrides)