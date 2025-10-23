#!/usr/bin/env python3
from __future__ import annotations
import os, shutil, tempfile, subprocess, sys
from pathlib import Path

# Stability toggles (safe on all setups)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

MODEL_PATH = Path("artifacts/tf_manifest_only/model.keras")  # or .h5 / SavedModel dir
ONNX_OUT   = Path("artifacts/tf_manifest_only/onnx/unet1d_fp32.onnx")
OPSET      = 17

def main():
    import tensorflow as tf
    import tf2onnx

    # 1) Load your trained model (no recompile)
    base = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2) Wrap to convert between ONNX [B,1,T] and Keras [B,T,1]
    class IOLayoutWrapper(tf.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        @tf.function(input_signature=[
            tf.TensorSpec([None, 1, None], tf.float32, name="noisy")
        ])
        def __call__(self, x):
            # [B,1,T] -> [B,T,1]
            x = tf.transpose(x, [0, 2, 1])
            y = self.m(x, training=False)     # expect [B,T,1]
            y = tf.transpose(y, [0, 2, 1])    # [B,1,T]
            return {"enhanced": y}

    wrapped = IOLayoutWrapper(base)
    concrete = wrapped.__call__.get_concrete_function()

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

    # 3) Try converting the tf.function first (works across tf2onnx versions)
    try:
        # tf2onnx >= 1.9 exposes convert.from_function
        model_proto, _ = tf2onnx.convert.from_function(
            concrete,
            input_signature=[tf.TensorSpec([None, 1, None], tf.float32, name="noisy")],
            opset=OPSET,
            output_path=ONNX_OUT.as_posix(),
        )
        print(f"[OK] ONNX written (from_function): {ONNX_OUT}")
        return
    except Exception as e_fn:
        print(f"[WARN] from_function failed: {e_fn}")

    # 4) Fallback: export a temporary SavedModel then try from_saved_model (if available)
    tmp = Path(tempfile.mkdtemp(prefix="ulps_sm_"))
    try:
        tf.saved_model.save(
            wrapped,
            tmp.as_posix(),
            signatures={"serving_default": concrete},
        )

        # Some versions expose convert.from_saved_model, try it
        from_saved = getattr(tf2onnx.convert, "from_saved_model", None)
        if callable(from_saved):
            try:
                from_saved(
                    tmp.as_posix(),
                    opset=OPSET,
                    output_path=ONNX_OUT.as_posix(),
                )
                print(f"[OK] ONNX written (from_saved_model): {ONNX_OUT}")
                return
            except Exception as e_sm:
                print(f"[WARN] from_saved_model failed: {e_sm}")

        # 5) Last resort: call CLI (bundled with tf2onnx)
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", tmp.as_posix(),
            "--opset", str(OPSET),
            "--output", ONNX_OUT.as_posix(),
        ]
        print("[INFO] Falling back to CLI:", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"[OK] ONNX written (cli): {ONNX_OUT}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
