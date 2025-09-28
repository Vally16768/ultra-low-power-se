#!/usr/bin/env python3
import json, argparse, os, time, subprocess

p=argparse.ArgumentParser()
p.add_argument("--metrics", required=True)
p.add_argument("--exp-id", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--commit", default=os.getenv("GITHUB_SHA","unknown"))
args=p.parse_args()

with open(args.metrics) as f:
    m=json.load(f)

# Acceptă fie {"metrics":{...}}, fie direct {...}
metrics = m.get("metrics", m)

# Info rtf/latency/power pot fi umplute ulterior
enriched = {
  "exp_id": args.exp_id,
  "commit": args.commit[:7],
  "dataset": args.dataset,
  "fs_hz": 48000,
  "metrics": metrics,
  "latency_ms_stream": metrics.get("latency_ms_stream", None),
  "rtf": metrics.get("rtf", None),
  "power_mw_avg": metrics.get("power_mw_avg", None),
  "energy_mj_per_s": metrics.get("energy_mj_per_s", None),
  "passed_gates": None
}

# Înlocuiește fișierul
with open(args.metrics,"w") as f:
    json.dump(enriched, f, indent=2)
print("[augment_metrics] wrote", args.metrics)
