#!/usr/bin/env python3
import argparse, json, os

ap=argparse.ArgumentParser()
ap.add_argument("--fp32", required=True)
ap.add_argument("--int8", default=None)
ap.add_argument("--out", required=True)
args=ap.parse_args()

def load(p):
    if p and os.path.exists(p):
        j=json.load(open(p))
        return j.get("metrics", j)
    return {}

fp=load(args.fp32)
q8=load(args.int8)

def row(k, title):
    a = fp.get(k,"-")
    b = q8.get(k,"-") if q8 else "-"
    d = (float(b)-float(a)) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else "-"
    return f"<tr><td>{title}</td><td>{a}</td><td>{b}</td><td>{d}</td></tr>"

html=f"""<!doctype html><html><head>
<meta charset="utf-8"><title>SE Report</title>
<style>body{{font-family:sans-serif}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body>
<h1>Speech Enhancement – Metrics</h1>
<table>
<tr><th>Metric</th><th>FP32</th><th>INT8</th><th>Δ(INT8-FP32)</th></tr>
{row('pesq_wb','PESQ(wb)')}
{row('stoi','STOI')}
{row('delta_snr_db','ΔSNR (dB)')}
{row('si_sdr_db','SI-SDR (dB)')}
</table>
<p>Note: Δ close to 0 ⇒ paritate bună.</p>
</body></html>
"""
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out,"w").write(html)
print("[report] wrote", args.out)
