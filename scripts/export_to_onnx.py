#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, sys, os, shutil, tempfile, subprocess
import tensorflow as tf

# Defaults for stable conversions
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

def build_wrapped_model(base, layout: str, fixed_len: int | None):
    """
    Wrap Keras model so ONNX sees a stable signature and (optionally) a fixed T.
    - channels_last : ONNX I/O [B, T, 1]  (default for our toolbox)
    - channels_first: ONNX I/O [B, 1, T]
    """
    assert layout in ("channels_last", "channels_first")
    time_dim = fixed_len if fixed_len else None

    if layout == "channels_last":
        @tf.function(input_signature=[
            tf.TensorSpec([None, time_dim, 1], tf.float32, name="input")
        ])
        def serving_fn(x):
            # x: [B,T,1] -> model expects [B,T,1]
            y = base(x, training=False)
            # Ensure 3D output
            if len(y.shape) == 2:
                y = y[..., None]
            return {"output": y}
        return serving_fn

    else:  # channels_first
        @tf.function(input_signature=[
            tf.TensorSpec([None, 1, time_dim], tf.float32, name="input")
        ])
        def serving_fn(x):
            # convert to [B,T,1]
            x = tf.transpose(x, [0, 2, 1])
            y = base(x, training=False)
            if len(y.shape) == 2:
                y = y[..., None]
            # back to [B,1,T]
            y = tf.transpose(y, [0, 2, 1])
            return {"output": y}
        return serving_fn

def try_from_function(serving_fn, out_path: pathlib.Path, opset: int) -> bool:
    try:
        import tf2onnx
        # Pass the tf.function itself (not ConcreteFunction) for older tf2onnx
        tf2onnx.convert.from_function(
            serving_fn,
            input_signature=serving_fn.input_signature,
            opset=opset,
            output_path=out_path.as_posix(),
        )
        print(f"[OK] ONNX written (from_function): {out_path}")
        return True
    except Exception as e:
        print(f"[WARN] from_function failed: {e}")
        return False

def try_from_savedmodel(serving_fn, out_path: pathlib.Path, opset: int) -> bool:
    import tf2onnx
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ulps_sm_"))
    try:
        conc = serving_fn.get_concrete_function()
        tf.saved_model.save(
            obj=serving_fn.python_function.__self__
                if hasattr(serving_fn.python_function, "__self__") else serving_fn,
            export_dir=tmp.as_posix(),
            signatures={"serving_default": conc},
        )
        from_saved = getattr(tf2onnx.convert, "from_saved_model", None)
        if callable(from_saved):
            try:
                from_saved(tmp.as_posix(), opset=opset, output_path=out_path.as_posix())
                print(f"[OK] ONNX written (from_saved_model): {out_path}")
                return True
            except Exception as e:
                print(f"[WARN] from_saved_model failed: {e}")

        # CLI fallback
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", tmp.as_posix(),
            "--opset", str(opset),
            "--output", out_path.as_posix(),
        ]
        print("[INFO] Falling back to CLI:", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"[OK] ONNX written (cli): {out_path}")
        return True
    except Exception as e:
        print(f"[WARN] SavedModel path failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    p = argparse.ArgumentParser("Export Keras/SavedModel to ONNX (stable)")
    p.add_argument("--src",  type=str, required=True,
                   help="Path to model.keras/model.h5 or SavedModel dir")
    p.add_argument("--out",  type=str, required=True,
                   help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fixed_len", type=int, default=None,
                   help="Optional fixed T (samples). If None, export dynamic length.")
    p.add_argument("--layout", choices=["channels_last","channels_first"],
                   default="channels_last",
                   help="ONNX I/O layout. 'channels_last' => [B,T,1] (default).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    if src.is_dir():
        base = tf.keras.models.load_model(src)
    else:
        base = tf.keras.models.load_model(src, compile=False)

    serving_fn = build_wrapped_model(base, args.layout, args.fixed_len)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if try_from_function(serving_fn, out_path, args.opset):
        return
    if try_from_savedmodel(serving_fn, out_path, args.opset):
        return

    raise SystemExit("ONNX export failed via all methods.")

if __name__ == "__main__":
    main()
