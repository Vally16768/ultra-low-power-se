# se_cli/cli.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny, robust CLI for Ultra-Low-Power SE.

Usage:
  python -m se_cli.cli <cmd> --config <file.yaml> [-o key.path=VALUE ...]
Commands:
  train | eval | enhance | score | export
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import inspect
from typing import Dict, List, Any

from se_cli.config import load_config

SUBCMDS = {"train", "eval", "enhance", "score", "export"}


def _get_runner(cmd: str):
    if cmd == "train":
        from runners import train as r
        return r.main
    if cmd == "eval":
        from runners import evaluate as r
        return r.main
    if cmd == "enhance":
        from runners import infer as r
        return r.main
    if cmd == "score":
        from runners import score as r
        return r.main
    if cmd == "export":
        from runners import export as r
        return r.main
    raise SystemExit(f"Comandă necunoscută: {cmd}")


def _flatten_overrides(raw: List[List[str]] | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for group in (raw or []):
        for item in group:
            if "=" not in item:
                raise SystemExit(f"Override invalid: {item}. Folosește key.path=VALUE")
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _coerce_value(val: str) -> Any:
    try:
        return json.loads(val)
    except Exception:
        pass
    low = val.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            return int(val)
        return float(val)
    except ValueError:
        return val


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, str]) -> None:
    for key, raw_val in overrides.items():
        cur: Dict[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]  # type: ignore[assignment]
        cur[parts[-1]] = _coerce_value(raw_val)


def _apply_model_module_env(cfg: Dict[str, Any]) -> None:
    mod = os.environ.get("MODEL_MODULE")
    if mod:
        cfg.setdefault("model", {})
        cfg["model"]["module"] = mod


def _runner_accepts_cfg(run) -> bool:
    try:
        sig = inspect.signature(run)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 1


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"Usage: python -m se_cli.cli <cmd> --config <file.yaml> [-o key.path=VALUE ...]")
        print(f"Comenzi: {', '.join(sorted(SUBCMDS))}")
        return 0 if argv else 1

    cmd, rest = argv[0], argv[1:]
    if cmd not in SUBCMDS:
        print(f"Comandă necunoscută: {cmd}. Opțiuni: {', '.join(sorted(SUBCMDS))}", file=sys.stderr)
        return 2

    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--config", required=True, help="Fișier YAML de config")
    g.add_argument("-o", "--override", nargs="+", action="append", default=[],
                   help="Override-uri key.path=VALUE (se pot repeta)")
    try:
        gargs, _unknown = g.parse_known_args(rest)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    if not os.path.isfile(gargs.config):
        print(f"[cli] Config inexistent: {gargs.config}", file=sys.stderr)
        return 2

    try:
        overrides_raw = _flatten_overrides(gargs.override)
        cfg = load_config(gargs.config, overrides_raw)
        _apply_overrides(cfg, overrides_raw)
        _apply_model_module_env(cfg)
    except Exception as exc:
        print(f"[cli] Eroare la încărcarea configului: {exc}", file=sys.stderr)
        return 2

    run = _get_runner(cmd)

    if _runner_accepts_cfg(run):
        try:
            ret = run(cfg)
            return int(ret) if isinstance(ret, int) else 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0
        except Exception as exc:
            print(f"[cli] Runner '{cmd}' a eșuat: {exc}", file=sys.stderr)
            return 1
    else:
        os.environ["SE_CONFIG_PATH"] = gargs.config
        os.environ["SE_CONFIG_OVERRIDES_JSON"] = json.dumps(overrides_raw)
        try:
            ret = run()
            return int(ret) if isinstance(ret, int) else 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0
        except Exception as exc:
            print(f"[cli] Runner '{cmd}' a eșuat: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
