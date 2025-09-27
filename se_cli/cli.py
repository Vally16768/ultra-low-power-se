import sys, argparse
from se_cli.config import load_config

# import lazily ca să nu crape dacă lipsesc temporar
def _get_runner(cmd):
    if cmd == "train":
        from runners import train as r; return r.main
    if cmd == "eval":
        from runners import infer as r; return r.main
    if cmd == "enhance":
        from runners import infer as r; return r.main
    if cmd == "score":
        from runners import score as r; return r.main
    if cmd == "export":
        from runners import export as r; return r.main
    raise SystemExit(f"Comandă necunoscută: {cmd}")

SUBCMDS = {"train","eval","enhance","export","score"}

def _split_on_subcmd(argv):
    """Împarte argv în (globals, cmd, rest) indiferent de poziție."""
    for i, tok in enumerate(argv):
        if tok in SUBCMDS:
            return argv[:i], tok, argv[i+1:]
    # dacă nu găsește subcomandă, lasă argparse să dea eroare
    return argv, None, []

def _parse_globals(args):
    gp = argparse.ArgumentParser(add_help=False)
    gp.add_argument("--config", required=True, help="configs/<exp>.yaml")
    # Acceptă una sau mai multe valori la un singur -o/--override
    gp.add_argument("-o","--override", nargs="+", action="append", default=[],
                    help="override-uri key=value; poți pune mai multe după un -o")
    return gp.parse_args(args)

def _flatten_overrides(ov):
    # ov este listă de liste (din action=append, nargs='+')
    flat = []
    for group in ov:
        flat.extend(group)
    # fă dict key=value
    d = {}
    for kv in flat:
        if "=" in kv:
            k,v = kv.split("=",1); d[k]=v
    return d

def main(argv=None):
    argv = list(argv or sys.argv[1:])
    # scoatem help global rapid
    if not argv or argv == ["-h"] or argv == ["--help"]:
        p = argparse.ArgumentParser(prog="se")
        p.add_argument("--config", required=True)
        p.add_argument("-o","--override", nargs="+", action="append", default=[])
        p.add_argument("cmd", choices=sorted(SUBCMDS))
        p.print_help(); sys.exit(0)

    g_before, cmd, g_after = _split_on_subcmd(argv)
    if cmd is None:
        # nu s-a găsit subcomanda; forțăm argparse să dea mesaj clar
        p = argparse.ArgumentParser(prog="se")
        p.add_argument("--config", required=True)
        p.add_argument("-o","--override", nargs="+", action="append", default=[])
        p.add_argument("cmd", choices=sorted(SUBCMDS))
        p.parse_args(argv)  # va arunca eroare cu mesaj util
        return

    gargs = _parse_globals(g_before + g_after)  # permite --config/-o oriunde
    overrides = _flatten_overrides(gargs.override)
    cfg = load_config(gargs.config, overrides)

    run = _get_runner(cmd)
    return run(cfg)

if __name__ == "__main__":
    main()