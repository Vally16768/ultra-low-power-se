# parity_check.py
#!/usr/bin/env python3
import argparse, numpy as np, onnxruntime as ort, tensorflow as tf, soundfile as sf

def pick_io(sess):
    in0 = sess.get_inputs()[0].name
    out0 = sess.get_outputs()[0].name
    return in0, out0

def load_tf(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def run_tf(m, xin_btc):
    # Keras model expects [B,T,1]; keep as is
    y = m(xin_btc, training=False)
    return y.numpy()

def run_ort(sess, input_name, xin_btc):
    return sess.run([sess.get_outputs()[0].name], {input_name: xin_btc})[0]

def snr_db(ref, err, eps=1e-12):
    num = np.sum(ref**2) + eps
    den = np.sum(err**2) + eps
    return 10*np.log10(num/den)

def main():
    ap = argparse.ArgumentParser("Compare TF and ONNX outputs on random or file")
    ap.add_argument("--tf_model", default="artifacts/tf_manifest_only/model.keras")
    ap.add_argument("--onnx", default="artifacts/tf_manifest_only/onnx/unet1d_fp32.onnx")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--wav", type=str, default=None, help="Optional .wav instead of random")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    if args.wav:
        x, sr = sf.read(args.wav)
        assert sr == args.sr, f"Expected {args.sr} Hz, got {sr}"
        if x.ndim == 2: x = x.mean(axis=1)
        x = x.astype(np.float32)
    else:
        n = int(args.seconds * args.sr)
        x = (np.random.randn(n).astype(np.float32)) * 0.1

    xin = x[None, :, None]  # [1,T,1]

    tf_model = load_tf(args.tf_model)
    y_tf = run_tf(tf_model, xin)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name, out_name = pick_io(sess)
    y_onnx = sess.run([out_name], {in_name: xin})[0]

    mae = float(np.mean(np.abs(y_tf - y_onnx)))
    mse = float(np.mean((y_tf - y_onnx)**2))
    refsnr = float(snr_db(y_tf, y_tf - y_onnx))
    print(f"Parity  MAE: {mae:.6e}  MSE: {mse:.6e}  SNR(ref vs err): {refsnr:.2f} dB")

    import os
    outdir = os.path.join("artifacts", "onnx_parity")
    os.makedirs(outdir, exist_ok=True)
    sf.write(os.path.join(outdir, "input.wav"), x, args.sr)
    sf.write(os.path.join(outdir, "tf_out.wav"),  y_tf.squeeze(), args.sr)
    sf.write(os.path.join(outdir, "onnx_out.wav"), y_onnx.squeeze(), args.sr)
    print(f"[Saved] {outdir}/(input.wav, tf_out.wav, onnx_out.wav)")

if __name__ == "__main__":
    main()
