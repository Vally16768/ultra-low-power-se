# eval/bench_onnx.py
# Bench calitate & viteză pentru ONNX — fără dependențe externe
import argparse, json, time, random
import numpy as np
from pathlib import Path

def si_snr(ref, est, eps=1e-8):
    # scale-invariant SNR (Le Roux et al.)
    import numpy as np
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    # proiecție optimă: est_hat = <est, ref> / ||ref||^2 * ref
    dot = np.sum(est * ref)
    ref_energy = np.sum(ref * ref) + eps
    scale = dot / ref_energy
    est_hat = scale * ref
    e_noise = est - est_hat
    return 10 * np.log10((np.sum(est_hat * est_hat) + eps) / (np.sum(e_noise * e_noise) + eps))

def delta_si_snr(ref, noisy, est):
    return float(si_snr(ref, est) - si_snr(ref, noisy))


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    if x.ndim == 2:  # (T, C)
        return x.mean(axis=1)
    return x

def _resample_linear(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    n_out = int(round(len(x) * (fs_out / fs_in)))
    if n_out <= 1 or len(x) == 0:
        return np.zeros((0,), dtype=x.dtype)
    t_in  = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype="float64")
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype="float64")
    return np.interp(t_out, t_in, x.astype("float64")).astype("float32")

def _sanitize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    m = ~np.isfinite(arr)
    if m.any(): arr[m] = 0.0
    return arr

def mix(clean, noise, snr_db=5.0):
    cs = clean / (np.std(clean) + 1e-9)
    ns = noise / (np.std(noise) + 1e-9)
    alpha = 10 ** (-snr_db / 20)
    x = cs + alpha * ns
    return _sanitize(x), _sanitize(cs)

def load_rand(paths, target_len, sr):
    import soundfile as sf
    p = random.choice(paths)
    x, fs = sf.read(p, always_2d=False)
    x = _to_mono(x)
    if fs != sr:
        x = _resample_linear(x, fs, sr)

    # dacă e mai scurt → pad; dacă e mai lung → decupare random
    if len(x) < target_len:
        x = np.pad(x, (0, target_len - len(x)))
        x = x[:target_len]
    elif len(x) > target_len:
        start = random.randint(0, len(x) - target_len)
        x = x[start:start + target_len]
    else:
        # exact
        pass

    return _sanitize(x)

def metric_pesq(ref, est, sr=16000):
    try:
        from pesq import pesq
        return float(pesq(sr, ref, est, 'wb'))
    except Exception:
        return None

def metric_stoi(ref, est, sr=16000):
    try:
        from pystoi import stoi
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return None

def snr_impr(ref, noisy, est):
    def snr(a, b): return 10*np.log10((np.sum(a*a)+1e-9)/(np.sum((a-b)**2)+1e-9))
    return float(snr(ref, est) - snr(ref, noisy))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--noise-list", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--metrics", nargs="+", default=["pesq","stoi","snr"])
    ap.add_argument("--latency-runs", type=int, default=100)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=0, help="intra_op_num_threads; 0=ORT default")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    import onnxruntime as ort
    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name; outp = sess.get_outputs()[0].name

    clean_paths = Path(args.clean_list).read_text().strip().splitlines()
    noise_paths = Path(args.noise_list).read_text().strip().splitlines()
    L = int(args.sr * args.secs)

    scores = {k: [] for k in args.metrics}
    lat = []

    # warmup
    dummy = np.zeros((1,1,L), dtype="float32")
    for _ in range(args.warmup):
        _ = sess.run([outp], {inp: dummy})

    for _ in range(args.n):
        c = load_rand(clean_paths, L, args.sr)
        n = load_rand(noise_paths, L, args.sr)
        noisy, clean = mix(c, n, snr_db=random.choice([-5, 0, 5, 10]))
        x = noisy[None, None, :].astype("float32")
        t0 = time.time()
        y = sess.run([outp], {inp: x})[0].squeeze().astype("float32")
        lat.append((time.time()-t0)*1000.0)

        if "pesq" in scores:
            v = metric_pesq(clean, y, sr=args.sr)
            if v is not None: scores["pesq"].append(v)
        if "stoi" in scores:
            v = metric_stoi(clean, y, sr=args.sr)
            if v is not None: scores["stoi"].append(v)
        if "snr" in scores:
            scores["snr"].append(snr_impr(clean, noisy, y))
        if "sisnr" in scores:
            scores["sisnr"].append(delta_si_snr(clean, noisy, y))

    out = {
        "model": args.model,
        "N": args.n,
        "metrics": {k: {"mean": float(np.mean(v)) if v else None,
                        "std": float(np.std(v)) if v else None,
                        "n": len(v)} for k,v in scores.items()},
        "latency_ms": {"mean": float(np.mean(lat)),
                       "p50": float(np.percentile(lat,50)),
                       "p90": float(np.percentile(lat,90)),
                       "p99": float(np.percentile(lat,99))}
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
