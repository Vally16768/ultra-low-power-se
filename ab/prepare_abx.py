#!/usr/bin/env python3
import argparse, os, shutil, glob, random, json

ap=argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--est", required=True)
ap.add_argument("--out", required=True)
args=ap.parse_args()

os.makedirs(args.out, exist_ok=True)
pairs=[]
ref_map={os.path.basename(p):p for p in glob.glob(os.path.join(args.ref,"**","*.wav"), recursive=True)}
for est in glob.glob(os.path.join(args.est,"**","*.wav"), recursive=True):
    base=os.path.basename(est)
    if base in ref_map:
        pairs.append((ref_map[base], est, base))
random.seed(0)
random.shuffle(pairs)

ab_dir=os.path.join(args.out,"abx")
os.makedirs(ab_dir, exist_ok=True)
manifest=[]
for i,(r,e,name) in enumerate(pairs[:50]):   # max 50 perechi
    d=os.path.join(ab_dir, f"case_{i:03d}")
    os.makedirs(d, exist_ok=True)
    A,B = (r,e) if random.random()<0.5 else (e,r)
    shutil.copy2(A, os.path.join(d,"A.wav"))
    shutil.copy2(B, os.path.join(d,"B.wav"))
    shutil.copy2(r, os.path.join(d,"ref.wav"))
    manifest.append({"case":i,"ref":os.path.relpath(r,args.out),"est":os.path.relpath(e,args.out)})
json.dump({"pairs":manifest}, open(os.path.join(args.out,"manifest.json"),"w"), indent=2)
print("[abx] wrote", args.out, "cases:", len(manifest))
