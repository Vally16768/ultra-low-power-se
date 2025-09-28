import sys, argparse
from typing import Dict, List
from se_cli.config import load_config

SUBCMDS = {"train","eval","enhance","score","export"}

def _get_runner(cmd):
    if cmd == "train":
        from runners import train as r; return r.main
    if cmd == "eval":
        from runners import evaluate as r; return r.main
    if cmd == "enhance":
        from runners import infer as r; return r.main
    if cmd == "score":
        from runners import score as r; return r.main
    if cmd == "export":
        from runners import export as r; return r.main
    raise SystemExit(f"Comandă necunoscută: {cmd}")

def _flatten_overrides(raw: List[List[str]] | None) -> Dict[str, str]:
    out: Dict[str,str] = {}
    for group in (raw or []):
        for item in group:
            if "=" not in item:
                raise SystemExit(f"Override invalid: {item}. Folosește key.path=VALUE")
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main(argv: List[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m se_cli.cli <cmd> --config <file.yaml> [-o key=val ...]")
        print(f"Comenzi: {', '.join(sorted(SUBCMDS))}")
        return 1

    cmd = argv[0]; rest = argv[1:]
    if cmd not in SUBCMDS:
        raise SystemExit(f"Comandă necunoscută: {cmd}. Opțiuni: {', '.join(sorted(SUBCMDS))}")

    # parse globals anywhere
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--config", required=True)
    g.add_argument("-o","--override", nargs="+", action="append", default=[])
    gargs, _ = g.parse_known_args(rest)
    overrides = _flatten_overrides(gargs.override)

    cfg = load_config(gargs.config, overrides)
    run = _get_runner(cmd)
    return run(cfg)

if __name__ == "__main__":
    main()