# se_cli/cli.py
import sys, importlib, yaml, argparse
from typing import Dict, List, Any

SUBCMDS = {"train","eval","enhance","score","export"}

# se_cli/cli.py – înlocuiește helper-ele de parsing/merge
import yaml

def _parse_value(v: str):
    # încearcă YAML pentru tipare (bool, int, float, list, dict)
    try:
        return yaml.safe_load(v)
    except Exception:
        return v

def _flatten_overrides(raw):
    out = {}
    for group in (raw or []):
        for item in group:
            if "=" not in item:
                raise SystemExit(f"Override invalid: {item}. Folosește key.path=VALUE")
            k, v = item.split("=", 1)
            out[k.strip()] = _parse_value(v.strip())
    return out

def _merge(d, path, value):
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _get_runner(cmd):
    mod = {
        "train": "runners.train",
        "eval": "runners.evaluate",
        "enhance": "runners.infer",
        "score": "runners.score",
        "export": "runners.export",
    }[cmd]
    return importlib.import_module(mod).main

def _load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _apply_overrides(cfg: Dict[str, Any], ov: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in ov.items():
        _merge(cfg, k, v)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(SUBCMDS))
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", nargs="+",
                    help="override-uri key.path=VALUE (poți repeta)")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    ovs = _flatten_overrides(args.override)
    cfg = _apply_overrides(cfg, ovs)

    runner = _get_runner(args.cmd)
    return runner(cfg)

if __name__ == "__main__":
    main()
