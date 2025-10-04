from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os, re

_DOLLAR = re.compile(r"^\$\{([^}]+)\}$")

def _get_by_path(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def resolve_pathlike(val: Any, cfg: Dict[str, Any]) -> str | None:
    """
    Expandează:
      - placeholderul '${a.b.c}' din cfg (dotted path)
      - '~' și variabile de mediu
    Returnează cale normalizată sau None dacă val nu e string.
    """
    if not isinstance(val, str):
        return None
    m = _DOLLAR.match(val.strip())
    if m:
        ref = _get_by_path(cfg, m.group(1))
        if isinstance(ref, str):
            val = ref
    val = os.path.expandvars(os.path.expanduser(val))
    return str(Path(val))
