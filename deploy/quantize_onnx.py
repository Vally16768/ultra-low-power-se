# deploy/quantize_onnx.py
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--weight_type", choices=["QInt8","QUInt8"], default="QInt8")
    args = ap.parse_args()
    wt = QuantType.QInt8 if args.weight_type=="QInt8" else QuantType.QUInt8
    quantize_dynamic(args.in_model, args.out_model, weight_type=wt)
    print(f"[quantize] wrote {args.out_model}")

if __name__ == "__main__":
    main()
