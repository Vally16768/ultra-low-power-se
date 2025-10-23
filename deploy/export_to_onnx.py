#!/usr/bin/env python3
import argparse, pathlib, sys
import tensorflow as tf
import tf2onnx

def main():
    p = argparse.ArgumentParser("Export Keras/SavedModel to ONNX")
    p.add_argument("--src", type=str, default="artifacts/tf_manifest_only/model.keras",
                   help="Path to model.keras, model.h5, or SavedModel dir")
    p.add_argument("--out", type=str, default="artifacts/tf_manifest_only/model.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fixed_len", type=int, default=None,
                   help="Optional fixed number of samples (e.g. 32000). If None, export dynamic time.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    if src.is_dir():
        model = tf.keras.models.load_model(src)  # SavedModel
    else:
        model = tf.keras.models.load_model(src, compile=False)

    # Create a concrete function with dynamic or fixed time dimension
    @tf.function(input_signature=[tf.TensorSpec([None, args.fixed_len if args.fixed_len else None, 1],
                                               tf.float32, name="input")])
    def serving_fn(x):
        return {"output": model(x, training=False)}

    concrete = serving_fn.get_concrete_function()

    onnx_model, _ = tf2onnx.convert.from_function(
        concrete_function=concrete,
        opset=args.opset,
        input_names=["input:0"],
        output_names=["output:0"],
        large_model=False
    )
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    if args.verbose:
        print(f"[OK] Exported to: {out_path.resolve()} (opset={args.opset}, fixed_len={args.fixed_len})")

if __name__ == "__main__":
    main()
