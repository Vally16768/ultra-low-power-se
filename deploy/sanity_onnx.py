#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
import itertools as it

import onnx


def _shape_str(shape) -> str:
    dims: List[str] = []
    for d in shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value is not None:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return "[" + ",".join(dims) + "]"


def sanity_report(onnx_path: str) -> Dict[str, Any]:
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    opset = None
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx"):
            opset = int(o.version)

    inputs = []
    for v in list(m.graph.input):
        t = v.type.tensor_type
        inputs.append({"name": v.name, "shape": _shape_str(t.shape), "dtype": t.elem_type})

    outputs = []
    for v in list(m.graph.output):
        t = v.type.tensor_type
        outputs.append({"name": v.name, "shape": _shape_str(t.shape), "dtype": t.elem_type})

    from collections import Counter
    ops = Counter([n.op_type for n in m.graph.node])

    def _dyn_axes_for_valueinfo(vi) -> List[int]:
        dims = vi.type.tensor_type.shape.dim
        return [i for i, d in enumerate(dims) if bool(d.dim_param)]

    dyn_axes = {}
    for v in it.chain(list(m.graph.input), list(m.graph.output)):
        dyn_axes[v.name] = _dyn_axes_for_valueinfo(v)

    return {
        "ir_version": int(m.ir_version),
        "opset": opset,
        "num_nodes": len(m.graph.node),
        "num_initializers": len(m.graph.initializer),
        "ops_hist": dict(sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))),
        "io": {"inputs": inputs, "outputs": outputs},
        "dynamic_axes": dyn_axes,
        "ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", help="path la modelul .onnx")
    ap.add_argument("--print-json", dest="print_json", action="store_true", help="afișează JSON-ul complet")
    ap.add_argument("--min-opset", type=int, default=17, help="opset minim acceptat")
    args = ap.parse_args()

    rep = sanity_report(args.onnx_path)

    sidecar_path = Path(args.onnx_path + ".json")
    if sidecar_path.exists():
        try:
            rep["sidecar"] = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if rep["opset"] is not None:
        assert rep["opset"] >= args.min_opset, f"opset prea mic: {rep['opset']} < {args.min_opset}"
    has_dyn = any(len(ax) > 0 for ax in rep["dynamic_axes"].values())
    assert has_dyn, "nu s-au detectat axe dinamice (dim_param) — verifică dynamic_axes la export"

    if args.print_json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"[sanity] ONNX ok | opset={rep['opset']} | nodes={rep['num_nodes']} | dyn_axes={rep['dynamic_axes']}")
        top_ops = ", ".join(f"{k}:{v}" for k, v in list(rep["ops_hist"].items())[:10])
        print(f"[sanity] top ops: {top_ops}")


if __name__ == "__main__":
    main()
