#!/usr/bin/env python3
"""
Unified dataset builder — AGGREGATOR ONLY (no augmentation here)

- Citește VoiceBank-DEMAND (train/test) din data/voicebank-demand-16k.
- Colectează perechile (noisy, clean) din manifestele EXISTENTE sub data/prepared/**,
  scrise anterior de `make datasets` / mixgen.py.
- Scrie în data/datasets/final/:
    vbd_train.csv, vbd_test.csv, aug.csv, train.csv (=VBD∪AUG), val.csv (=VBD test), meta.json
"""

import csv, glob, json
from pathlib import Path

# ---------- helpers ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def discover_voicebank(vbd_dir: Path):
    def fpat(*pats):
        for pat in pats:
            f = sorted(glob.glob(str(vbd_dir / pat)))
            if f: return Path(f[0])
        return None
    trC = fpat("clean_trainset*_16k","clean_trainset*_wav","clean_trainset*")
    trN = fpat("noisy_trainset*_16k","noisy_trainset*_wav","noisy_trainset*")
    teC = fpat("clean_testset*_16k","clean_testset*_wav","clean_testset*")
    teN = fpat("noisy_testset*_16k","noisy_testset*_wav","noisy_testset*")
    if not all([trC,trN,teC,teN]):
        raise RuntimeError(f"[VBD] structura nevalidă în {vbd_dir}")
    return trN, trC, teN, teC

def pair_by_name(noisy_dir: Path, clean_dir: Path):
    nd = {Path(p).name: p for p in glob.glob(str(noisy_dir/"**/*.wav"), recursive=True)}
    cd = {Path(p).name: p for p in glob.glob(str(clean_dir/"**/*.wav"), recursive=True)}
    common = sorted(set(nd) & set(cd))
    return [(nd[k], cd[k]) for k in common]

def read_pairs_from_csv(csv_path: Path):
    """
    Acceptă capete flexibile; returnează list[(noisy, clean)].
    Suportă coloane: noisy/mixture/mix, respectiv clean/target/speech.
    Ignoră rânduri invalide.
    """
    pairs, rows = [], []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows: return pairs
    header = [h.strip().lower() for h in rows[0]]
    body = rows[1:] if any(h.isalpha() for h in header) else rows

    def idx(cands):
        for c in cands:
            if c in header: return header.index(c)
        return None

    i_noisy = idx(["noisy","mixture","mixture_path","noisy_path","mix","mix_path"])
    i_clean = idx(["clean","target","clean_path","speech","speech_path"])
    if i_noisy is None or i_clean is None:
        # fallback: primele două coloane
        i_noisy, i_clean = 0, 1
        body = rows

    for row in body:
        if len(row) <= max(i_noisy, i_clean): continue
        n, c = row[i_noisy].strip(), row[i_clean].strip()
        if n and c: pairs.append((n, c))
    return pairs

def collect_prepared_pairs(prepared_root: Path):
    """
    Caută manifestele în data/prepared/** și agregă perechile.
    Acceptă fișiere: manifest*.csv, pairs*.csv, mixes*.csv.
    """
    candidates = []
    for pat in ["**/manifest*.csv", "**/pairs*.csv", "**/mixes*.csv"]:
        candidates += glob.glob(str(prepared_root / pat), recursive=True)

    pairs, seen = [], set()
    for c in sorted(set(candidates)):
        p = Path(c)
        try:
            pr = read_pairs_from_csv(p)
        except Exception:
            continue
        for tup in pr:
            if tup in seen: continue
            seen.add(tup)
            pairs.append(tup)
    return pairs

# ---------- main ----------
def main():
    proj = Path.cwd()
    data = proj/"data"
    out_root = data/"datasets/final"
    ensure_dir(out_root)

    # 1) VoiceBank
    vbd_dir = data/"voicebank-demand-16k"
    trN, trC, teN, teC = discover_voicebank(vbd_dir)
    vbd_train = pair_by_name(trN, trC)
    vbd_test  = pair_by_name(teN, teC)

    # scrie CSV-urile VBD
    with open(out_root/"vbd_train.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(vbd_train)
    with open(out_root/"vbd_test.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(vbd_test)

    # 2) AUG din data/prepared/**
    prepared = data/"prepared"
    aug_pairs = collect_prepared_pairs(prepared) if prepared.is_dir() else []

    # 3) scrie CSV-urile finale
    with open(out_root/"aug.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(aug_pairs)

    final_train = vbd_train + aug_pairs
    final_val   = vbd_test
    with open(out_root/"train.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(final_train)
    with open(out_root/"val.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["noisy","clean"]); w.writerows(final_val)

    meta = {
        "mode": "prepared",
        "counts": {
            "vbd_train": len(vbd_train),
            "vbd_test":  len(vbd_test),
            "aug": len(aug_pairs),
            "final_train": len(final_train),
            "final_val": len(final_val),
        },
        "paths": {
            "vbd_train_csv": str(out_root/"vbd_train.csv"),
            "vbd_test_csv":  str(out_root/"vbd_test.csv"),
            "aug_csv":       str(out_root/"aug.csv"),
            "final_train_csv": str(out_root/"train.csv"),
            "final_val_csv":   str(out_root/"val.csv"),
        },
    }
    with open(out_root/"meta.json","w") as f: json.dump(meta, f, indent=2)

    print("\n[SUMMARY]")
    for k,v in meta["counts"].items():
        print(f"  {k:12s}: {v}")
    print("  out:", meta["paths"]["final_train_csv"], "|", meta["paths"]["final_val_csv"])

if __name__ == "__main__":
    main()
