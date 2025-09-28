#!/usr/bin/env python3
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", help="Path to exported ONNX model")
    ap.add_argument("--frames", type=int, default=16000)
    args = ap.parse_args()

    onnx_p = Path(args.onnx_path)
    sess = ort.InferenceSession(onnx_p.as_posix(), providers=['CPUExecutionProvider'])

    # autodetect input/output names
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    T = args.frames
    x = np.random.randn(1,1,T).astype(np.float32)
    y = sess.run([out_name], {in_name: x})[0]
    print(f"[sanity] ok. input={in_name} out={out_name} | out-shape={y.shape}")

if __name__ == "__main__":
    main()
