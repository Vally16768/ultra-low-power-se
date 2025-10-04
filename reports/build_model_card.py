#!/usr/bin/env python3
import argparse, json, os, platform, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", required=True)
ap.add_argument("--timing", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

m = json.load(open(args.metrics))
t = json.load(open(args.timing))


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=10).strip()
    except Exception:
        return "n/a"


pyver = platform.python_version()
pip_freeze = sh("pip freeze | sed -n '1,100p'")

md = f"""# Model Card

**Exp:** `{m.get('exp_id')}`
**Commit:** `{m.get('commit')}`
**Dataset:** `{m.get('dataset')}`
**FS:** {m.get('fs_hz')} Hz

## Objective metrics
- PESQ(wb): **{m['metrics'].get('pesq_wb','n/a')}**
- STOI: **{m['metrics'].get('stoi','n/a')}**
- ΔSNR(dB): **{m['metrics'].get('delta_snr_db','n/a')}**
- SI-SDR(dB): **{m['metrics'].get('si_sdr_db','n/a')}**

## Runtime
- Audio seconds (evaluated): {t.get('audio_seconds','n/a')}
- RTF (approx offline): {t.get('rtf','n/a')}

## Environment
- Python: {pyver}
- First 100 lines of `pip freeze`:
{pip_freeze}

## Known limitations
- Streaming latency/power not measured in CI.
- Metrics may vary ± small tolerances.
"""
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(md)
print("[model_card] wrote", args.out)
