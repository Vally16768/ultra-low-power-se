#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml  # type: ignore[import-untyped]

from deploy.export_onnx import export_onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config pt. export")
    args = ap.parse_args()

    cfg_text = Path(args.config).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    export_onnx(cfg)


if __name__ == "__main__":
    main()
