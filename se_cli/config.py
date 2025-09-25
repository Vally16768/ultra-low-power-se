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
            # convert string "true/false/123/1e-3" la tipuri native când vine din CLI
            if isinstance(v, str):
                lv = v.lower().strip()
                if lv in ("true", "false"):
                    v = (lv == "true")
                else:
                    try:
                        if "." in lv or "e" in lv:
                            v = float(v)
                        else:
                            v = int(v)
                    except Exception:
                        pass
            out[k] = v
    return out

def kv_to_nested_dict(pairs: Dict[str, str]) -> Dict[str, Any]:
    """
    Transformă {"train.lr":"1e-3","model.name":"mamba_unet"} -> dict imbricat.
    """
    root: Dict[str, Any] = {}
    for full_k, v in (pairs or {}).items():
        cur = root
        keys = full_k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v
    return root

def load_config(config_path: str, override_kv: Dict[str, str] | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    overrides = kv_to_nested_dict(override_kv or {})
    return _merge_dicts(cfg, overrides)
