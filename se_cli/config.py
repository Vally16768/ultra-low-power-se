# se_cli/config.py
from __future__ import annotations
import re
from typing import Any, Dict
import yaml

_REF = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def _get_by_path(d: Dict[str, Any], path: str):
    cur: Any = d
    for k in path.split("."):
        cur = cur[k]
    return cur


def _resolve_refs(obj: Any, root: Dict[str, Any]):
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    if isinstance(obj, str):

        def repl(m):
            try:
                v = _get_by_path(root, m.group(1))
                return str(v)
            except Exception:
                return m.group(0)  # lasă nerezolvat dacă lipsește

        return _REF.sub(repl, obj)
    return obj


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # rezolvă interpolări simple ${...} (opțional pentru tine)
    cfg = _resolve_refs(cfg, cfg)
    return cfg
