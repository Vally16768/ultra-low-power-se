#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path


def collect_pairs_two_dirs(noisy, clean):
    pairs = []
    cidx = {p.stem: p for p in clean.rglob("*.wav")}
    for n in noisy.rglob("*.wav"):
        base = re.sub(r"([._-]noisy)$", "", n.stem)
        c = cidx.get(base) or cidx.get(n.stem)
        if c:
            pairs.append((str(n), str(c)))
    return pairs


def collect_pairs_same_dir(root):
    pairs = []
    idx = {p.stem: p for p in root.rglob("*.wav")}
    for n in list(idx.values()):
        base = re.sub(r"([._-]noisy)$", "", n.stem)
        c = idx.get(base + "_clean") or idx.get(base + "-clean") or idx.get(base + ".clean")
        if c:
            pairs.append((str(n), str(c)))
    return pairs


def write_manifest(pairs, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noisy", "clean"])
        w.writerows(pairs)
    print(f"[wrote] {out_csv} ({len(pairs)} pairs)")


def build_for(root, out_csv):
    if not root or not root.exists():
        return
    pairs = []
    for sub in [root] + [p for p in root.iterdir() if p.is_dir()]:
        nd, cd = sub / "noisy", sub / "clean"
        if nd.exists() and cd.exists():
            pairs += collect_pairs_two_dirs(nd, cd)
    pairs += collect_pairs_same_dir(root)
    seen = set()
    uniq = []
    for n, c in pairs:
        if (n, c) not in seen:
            uniq.append((n, c))
            seen.add((n, c))
    if uniq:
        write_manifest(uniq, out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path)
    ap.add_argument("--val-root", type=Path)
    ap.add_argument("--test-root", type=Path)
    a = ap.parse_args()
    if a.train_root:
        build_for(a.train_root, a.train_root / "manifests/pairs.csv")
    if a.val_root:
        build_for(a.val_root, a.val_root / "manifests/pairs.csv")
    if a.test_root:
        build_for(a.test_root, a.test_root / "manifests/pairs.csv")
