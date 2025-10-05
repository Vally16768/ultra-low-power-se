#!/usr/bin/env python3
import argparse, yaml, json
from typing import Dict, Any, List

def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _parse_overrides(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"Invalid override: {it}. Use key.path=value")
        path, val = it.split("=", 1)
        cur = out
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        try:
            cur[keys[-1]] = json.loads(val)
        except Exception:
            if val.lower() in ("true","false"):
                cur[keys[-1]] = (val.lower()=="true")
            else:
                cur[keys[-1]] = val
    return out

def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg

def main():
    import importlib
    ap = argparse.ArgumentParser("se_cli")
    ap.add_argument("cmd", choices=["train","eval","enhance","export","score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, _parse_overrides(args.override))
    mod = importlib.import_module(f"runners.{args.cmd if args.cmd!='enhance' else 'infer'}")
    mod.main(cfg)

if __name__ == "__main__":
    main()
