#!/usr/bin/env python3
import argparse, numpy as np, onnxruntime as ort, tensorflow as tf, soundfile as sf
from pathlib import Path

def load_tf(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def run_tf(m, x):
    return m(x, training=False).numpy()

def run_ort(sess, x):
    return sess.run([sess.get_outputs()[0].name], {"input:0": x})[0]

def snr_db(ref, err, eps=1e-12):
    num = np.sum(ref**2) + eps
    den = np.sum(err**2) + eps
    return 10*np.log10(num/den)

def main():
    ap = argparse.ArgumentParser("Compare TF and ONNX outputs on random or file")
    ap.add_argument("--tf_model", default="artifacts/tf_manifest_only/model.keras")
    ap.add_argument("--onnx", default="artifacts/tf_manifest_only/model.onnx")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--wav", type=str, default=None, help="Optional .wav to use instead of random")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    if args.wav:
        x, sr = sf.read(args.wav)
        assert sr == args.sr, f"Expected {args.sr} Hz, got {sr}"
        if x.ndim == 2: x = x.mean(axis=1)
        x = x.astype(np.float32)
    else:
        n = int(args.seconds * args.sr)
        x = np.random.randn(n).astype(np.float32) * 0.1

    # N, T, C
    xin = x[None, :, None]

    tf_model = load_tf(args.tf_model)
    y_tf = run_tf(tf_model, xin)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    y_onnx = run_ort(sess, xin)

    mae = np.mean(np.abs(y_tf - y_onnx))
    mse = np.mean((y_tf - y_onnx)**2)
    refsnr = snr_db(y_tf, y_tf - y_onnx)

    print(f"Parity MAE: {mae:.6e}  MSE: {mse:.6e}  SNR(ref vs err): {refsnr:.2f} dB")
    outdir = Path("artifacts/onnx_parity")
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "input.wav", x, args.sr)
    sf.write(outdir / "tf_out.wav", y_tf.squeeze(-1).squeeze(0), args.sr)
    sf.write(outdir / "onnx_out.wav", y_onnx.squeeze(-1).squeeze(0), args.sr)
    print(f"[Saved] {outdir}/(input.wav, tf_out.wav, onnx_out.wav)")

if __name__ == "__main__":
    main()
