#!/usr/bin/env python3
import json, sys, math, pathlib


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(baseline_path, metrics_path):
    base = load(baseline_path)
    cand = load(metrics_path)
    b = base["metrics"]
    a = cand.get("aggregate") or cand.get("metrics") or {}
    # toleranțe minime
    thr = {
        "PESQ": 3.00,
        "STOI": 0.93,
        "DELTA_SNR": 9.0,
        "SI_SDR": 12.0,
    }
    failures = []

    def get(key):
        return a.get(key) or a.get(key.lower()) or a.get(key.upper())

    for k, t in thr.items():
        v = get(k)
        if v is None:
            failures.append((k, "missing", t))
            continue
        if v < t:
            failures.append((k, v, t))
    print("[gate] candidate:", a)
    if failures:
        print("[gate] FAIL thresholds:", failures, file=sys.stderr)
        sys.exit(2)
    # compară și cu baseline (golden)
    drops = []
    for k, v_base in b.items():
        v = get(k) or get(k.replace("_db", "").upper())
        if v is None:
            continue
        # pesq/stoi cresc; dacă scade >0.02 => fail
        if k.lower() in ("pesq_wb", "pesq", "stoi") and v < v_base - 0.02:
            drops.append((k, v, v_base))
        # snr/sisdr cresc; dacă scade >0.2 dB => fail
        if k.lower() in ("delta_snr_db", "si_sdr_db", "delta_snr", "si_sdr") and v < v_base - 0.2:
            drops.append((k, v, v_base))
    if drops:
        print("[gate] FAIL vs baseline:", drops, file=sys.stderr)
        sys.exit(3)
    print("[gate] PASS ✅")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ci_gate.py BASELINE.json METRICS.json", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
