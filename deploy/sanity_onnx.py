#!/usr/bin/env python3
import argparse, onnx

def main():
    ap = argparse.ArgumentParser("Basic ONNX checks")
    ap.add_argument("--model", default="artifacts/tf_manifest_only/model.onnx")
    args = ap.parse_args()

    m = onnx.load(args.model)
    onnx.checker.check_model(m)
    g = m.graph
    print("[OK] Model loads & passes checker.")
    print("Inputs:")
    for i in g.input:
        t = i.type.tensor_type
        shape = [d.dim_param or d.dim_value for d in t.shape.dim]
        print(f" - {i.name}: {onnx.TensorProto.DataType.Name(t.elem_type)} {shape}")
    print("Outputs:")
    for o in g.output:
        t = o.type.tensor_type
        shape = [d.dim_param or d.dim_value for d in t.shape.dim]
        print(f" - {o.name}: {onnx.TensorProto.DataType.Name(t.elem_type)} {shape}")

if __name__ == "__main__":
    main()
