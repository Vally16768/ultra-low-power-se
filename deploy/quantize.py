# deploy/quantize.py
import argparse, os, random
from pathlib import Path


def seed_all(seed=404):
    import numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_list(p):
    return Path(p).read_text().strip().splitlines()


def load_audio_paths(clean_list, noisy_list, n):
    import itertools

    clean = _read_list(clean_list) if clean_list else []
    noisy = _read_list(noisy_list) if noisy_list else clean
    return list(itertools.islice(zip(clean, noisy), n))


def _to_mono(x):
    import numpy as np

    x = x.astype("float32", copy=False)
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def _resample_linear(x, fs_in, fs_out):
    import numpy as np

    if fs_in == fs_out:
        return x
    n_out = int(round(len(x) * (fs_out / fs_in)))
    if n_out <= 1 or len(x) == 0:
        return np.zeros((0,), dtype=x.dtype)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype="float64")
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype="float64")
    return np.interp(t_out, t_in, x.astype("float64")).astype("float32")


def make_calib_loader(pairs, sr=16000, secs=2.0):
    import numpy as np, soundfile as sf

    L = int(sr * secs)

    def gen():
        for c, _ in pairs:
            x, fs = sf.read(c, always_2d=False)
            x = _to_mono(x)
            if fs != sr:
                x = _resample_linear(x, fs, sr)
            if len(x) < L:
                x = np.pad(x, (0, L - len(x)))
            else:
                x = x[:L]
            yield x.astype("float32")[None, None, :]

    return gen


def _auto_boundary_nodes(onnx_path):
    # găsește primul și ultimul Conv din grafic (după topologie) pentru a le exclude din INT8
    import onnx

    model = onnx.load(onnx_path)
    nodes = [n for n in model.graph.node]
    conv_idxs = [i for i, n in enumerate(nodes) if n.op_type == "Conv"]
    if not conv_idxs:
        return []
    first_conv = nodes[conv_idxs[0]].name or nodes[conv_idxs[0]].output[0]
    last_conv = nodes[conv_idxs[-1]].name or nodes[conv_idxs[-1]].output[0]
    return [first_conv, last_conv]


def quantize_onnx_static(
    in_model,
    out_model,
    calib_pairs,
    keep_nodes=(),
    method="minmax",
    percentile=99.9,
    per_channel=True,
    activation_symmetric=False,
    weight_symmetric=True,
):
    import onnxruntime as ort
    from onnxruntime.quantization import (
        QuantType,
        CalibrationDataReader,
        quantize_static,
        CalibrationMethod,
        QuantFormat,
    )

    class CalibReader(CalibrationDataReader):
        def __init__(self, sess, gen_fn):
            self.sess = sess
            self.gen_fn = gen_fn
            self.enum = None
            self.input_name = sess.get_inputs()[0].name

        def get_next(self):
            if self.enum is None:
                self.enum = self.gen_fn()
            try:
                arr = next(self.enum)
            except StopIteration:
                return None
            # sanity: înlătură eventuale NaN/Inf din INPUT (rare)
            import numpy as np

            a = arr.astype("float32")
            a[~np.isfinite(a)] = 0.0
            return {self.input_name: a}

    sess = ort.InferenceSession(in_model, providers=["CPUExecutionProvider"])
    gen_fn = make_calib_loader(calib_pairs)

    from onnxruntime.quantization.calibrate import CalibrationMethod as CM

    cal_map = {"minmax": CM.MinMax, "entropy": CM.Entropy, "percentile": CM.Percentile}
    cal_method = cal_map.get(method.lower(), CM.MinMax)

    opts = {
        "quant_format": QuantFormat.QDQ,
        "activation_type": QuantType.QInt8 if activation_symmetric else QuantType.QUInt8,
        "weight_type": QuantType.QInt8 if weight_symmetric else QuantType.QUInt8,
        "per_channel": per_channel,
        "nodes_to_exclude": list(keep_nodes),
    }

    extra = {}
    if cal_method == CM.Percentile:
        extra = {"CalibratePercentile": float(percentile)}

    Path(os.path.dirname(out_model) or ".").mkdir(parents=True, exist_ok=True)

    def _run_quant(cal_meth, extra_opts):
        reader = CalibReader(sess, gen_fn)  # IMPORTANT: reader nou la fiecare rulare
        return quantize_static(
            model_input=in_model,
            model_output=out_model,
            calibration_data_reader=reader,
            calibrate_method=cal_meth,
            **opts,
            extra_options=extra_opts,
        )

    # încercare robustă + fallback (reconstruim reader la fiecare încercare)
    try:
        _run_quant(cal_method, extra)
    except Exception as e:
        msg = str(e)
        print(f"[quantize] WARN: '{method}' a eșuat ({msg}). Reîncerc cu MinMax…")
        _run_quant(CM.MinMax, {})
    return out_model


def quantize_onnx_dynamic(in_model, out_model, keep_nodes=(), per_channel=True, weight_symmetric=True):
    # Unele versiuni ORT CPU nu au kernel pt. ConvInteger (din QOperator dynamic).
    # Remediu: NU cuantizăm 'Conv' la dynamic; lăsăm doar MatMul (dacă există).
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from pathlib import Path
    import shutil, onnx

    Path(os.path.dirname(out_model) or ".").mkdir(parents=True, exist_ok=True)

    m = onnx.load(in_model)
    op_types = {n.op_type for n in m.graph.node}

    # dacă există doar Conv-uri, dynamic-quant ar produce ConvInteger (nesuportat) → skip elegant
    op_types_to_quantize = []
    if "MatMul" in op_types:
        op_types_to_quantize.append("MatMul")  # weight-only ok
    # nu adăuga "Conv" aici — altfel ajungem la ConvInteger

    if not op_types_to_quantize:
        # nimic de cuantizat în mod dinamic în siguranță → copiem inputul
        shutil.copy2(in_model, out_model)
        print("[PTQ-dynamic] no safe ops to quantize (Conv-only graph) → copied baseline.")
        return out_model

    quantize_dynamic(
        model_input=in_model,
        model_output=out_model,
        weight_type=QuantType.QInt8 if weight_symmetric else QuantType.QUInt8,
        per_channel=per_channel,
        nodes_to_exclude=list(keep_nodes),
        op_types_to_quantize=op_types_to_quantize,
    )
    return out_model


def export_qat_from_ckpt(ckpt, config, out_model, epochs=3, seed=404):
    seed_all(seed)
    import torch, yaml, inspect
    from importlib import import_module

    def load_factory():
        candidates = [
            ("se_models.mamba_unet.model", "build_model"),
            ("se_models.mamba_unet.model", "make_model"),
            ("se_models.mamba_unet.model", "create_model"),
            ("se_models.mamba_unet.model", "Model"),
            ("se_models.mamba_unet.model", "MambaUNet"),
        ]
        for mod, name in candidates:
            try:
                m = import_module(mod)
                return getattr(m, name)
            except Exception:
                pass
        raise ImportError("Nu am găsit un factory pentru model.")

    def construct_model(cfg_path):
        cfg = None
        if cfg_path:
            import yaml

            cfg = yaml.safe_load(Path(cfg_path).read_text())
        fn = load_factory()
        if inspect.isclass(fn):
            return fn.from_config(cfg) if (cfg and hasattr(fn, "from_config")) else (fn(**cfg) if cfg else fn())
        try:
            from inspect import signature

            sig = signature(fn)
            if cfg and len(sig.parameters) == 1:
                return fn(cfg)
            return fn(**cfg) if cfg else fn()
        except Exception:
            return fn()

    def export_to_onnx(model, out_path):
        dummy = torch.randn(1, 1, 32000)
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=18,
            input_names=["noisy"],
            output_names=["enhanced"],
            dynamic_axes={"noisy": {0: "B", 2: "T"}, "enhanced": {0: "B", 2: "T"}},
        )

    model = construct_model(config)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

    model.qconfig = get_default_qat_qconfig("fbgemm")
    model = prepare_qat(model.train())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dummy = torch.randn(4, 1, 32000)
    for _ in range(epochs):
        opt.zero_grad()
        y = model(dummy)
        loss = (y**2).mean()
        loss.backward()
        opt.step()
    model_int8 = convert(model.eval(), inplace=False)
    Path(os.path.dirname(out_model) or ".").mkdir(parents=True, exist_ok=True)
    export_to_onnx(model_int8, out_model)
    return out_model


def quantize_onnx_weight_only(in_model, out_model, keep_nodes=(), per_channel=True, weight_symmetric=True):
    # Convertim doar greutățile (Conv/MatMul) la INT8 prin QDQ, activările rămân FP32
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod, CalibrationDataReader
    import onnxruntime as ort

    Path(os.path.dirname(out_model) or ".").mkdir(parents=True, exist_ok=True)

    # Reader dummy cu un singur batch zero (nu calibrăm activări)
    class ZeroReader(CalibrationDataReader):
        def __init__(self, sess):
            import numpy as np

            self.done = False
            self.input_name = sess.get_inputs()[0].name
            self.shape = [1, 1, 32000]  # orice formă validă pentru modelul tău (din export)
            self.x = np.zeros(self.shape, dtype="float32")

        def get_next(self):
            if self.done:
                return None
            self.done = True
            return {self.input_name: self.x}

    sess = ort.InferenceSession(in_model, providers=["CPUExecutionProvider"])
    reader = ZeroReader(sess)

    quantize_static(
        model_input=in_model,
        model_output=out_model,
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.MinMax,
        # scoate linia cu activation_type=...
        weight_type=QuantType.QInt8 if weight_symmetric else QuantType.QUInt8,
        per_channel=per_channel,
        nodes_to_exclude=list(keep_nodes),
        quant_format=QuantFormat.QDQ,
        extra_options={"DisableActivationQuantization": True},  # dacă versiunea ta ORT o suportă
    )
    return out_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--calib-list", dest="calib_list", type=str)
    ap.add_argument("--noisy-list", dest="noisy_list", type=str)
    ap.add_argument("--num-calib", type=int, default=128)
    ap.add_argument("--keep-fp16-nodes", nargs="*", default=[], help="nume de noduri ONNX excluse de la INT8")
    ap.add_argument("--auto-keep-boundary", action="store_true", help="exclude automat primul și ultimul Conv")
    ap.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="percentile")
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--per-channel-weights", action="store_true", default=True)
    ap.add_argument("--no-per-channel-weights", dest="per_channel_weights", action="store_false")
    ap.add_argument("--activation-symmetric", action="store_true", default=False)
    ap.add_argument("--weight-asymmetric", action="store_true", default=False)
    # QAT
    ap.add_argument("--train-ckpt", type=str)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--config", type=str)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--weight-only", action="store_true", help="Conv/MatMul weights INT8, activations FP32")

    args = ap.parse_args()

    if args.qat:
        if not (args.train_ckpt and args.config):
            raise SystemExit("QAT cere --train-ckpt și --config")
        print(f"[QAT] {args.train_ckpt} → {args.out}")
        export_qat_from_ckpt(args.train_ckpt, args.config, args.out, epochs=args.epochs)
        print(f"[done] wrote: {args.out}")
        return

    if not args.model:
        raise SystemExit("PTQ cere --model (ONNX)")

    keep = list(args.keep_fp16_nodes)
    if args.auto_keep_boundary:
        keep += _auto_boundary_nodes(args.model)

    if args.static:
        if args.weight_only:
            print(f"[PTQ-weight-only] keep={keep} → {args.out}")
            quantize_onnx_weight_only(
                args.model, args.out, keep_nodes=tuple(keep), per_channel=args.per_channel_weights, weight_symmetric=not args.weight_asymmetric
            )
            print(f"[done] wrote: {args.out}")
            return

        if not args.calib_list:
            raise SystemExit("--calib-list necesar pentru PTQ static")
        pairs = load_audio_paths(args.calib_list, args.noisy_list, args.num_calib)
        print(f"[PTQ-static] calib={len(pairs)} method={args.method} pct={args.percentile} keep={keep} → {args.out}")
        quantize_onnx_static(
            args.model,
            args.out,
            pairs,
            keep_nodes=tuple(keep),
            method=args.method,
            percentile=args.percentile,
            per_channel=args.per_channel_weights,
            activation_symmetric=args.activation_symmetric,
            weight_symmetric=not args.weight_asymmetric,
        )
    elif args.dynamic:
        print(f"[PTQ-dynamic] keep={keep} → {args.out}")
        quantize_onnx_dynamic(args.model, args.out, keep_nodes=tuple(keep), per_channel=args.per_channel_weights, weight_symmetric=not args.weight_asymmetric)
    else:
        raise SystemExit("Specificați --static sau --dynamic (ori --qat)")

    print(f"[done] wrote: {args.out}")


if __name__ == "__main__":
    main()
