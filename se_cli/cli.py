import argparse
from se_cli.config import load_config, kv_to_nested_dict
from runners import train as run_train, evaluate as run_eval, enhance as run_enh, export as run_exp

def main():
    p = argparse.ArgumentParser("se")
    p.add_argument("--config", required=True, help="configs/<exp>.yaml")
    p.add_argument("-o","--override", action="append", help="key=value (dot.notation)", default=[])

    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("train"); sp.add_parser("eval"); sp.add_parser("enhance"); sp.add_parser("export")
    args = p.parse_args()

    # transformăm lista ["a.b=1","c=d"] într-un dict plat și apoi în nested
    flat: dict[str,str] = {}
    for kv in (args.override or []):
        k, v = kv.split("=", 1)
        flat[k] = v
    cfg = load_config(args.config, flat)

    if args.cmd == "train":   run_train.main(cfg)
    if args.cmd == "eval":    run_eval.main(cfg)
    if args.cmd == "enhance": run_enh.main(cfg)
    if args.cmd == "export":  run_exp.main(cfg)

if __name__ == "__main__":
    main()
