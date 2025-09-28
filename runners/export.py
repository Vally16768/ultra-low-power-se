from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from deploy.export_onnx import export_onnx

def main(cfg: Dict[str, Any]):
    out: Path = export_onnx(cfg)
    print(f"[export] ONNX saved to: {out}")
    return str(out)
