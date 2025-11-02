from __future__ import annotations
from typing import Optional, Union
import os
import torch

from .metrics_utils import finite_or_default, clamp, LOG

def _import_nisqa_class():
    try:
        from nisqa.NISQA_model import NISQA  # typical in repo
        return NISQA
    except Exception:
        pass
    try:
        from nisqa import NISQAModel as NISQA  # fallback alias
        return NISQA
    except Exception:
        raise ImportError(
            "Could not import NISQA. Ensure the NISQA package is installed "
            "from the official repo and available in PYTHONPATH."
        )

NISQA = _import_nisqa_class()

def load_nisqa(model_ref: str, device: Optional[Union[str, torch.device]] = None):
    """
    Load a NISQA model.

    model_ref:
      - a model-zoo name if your version supports from_pretrained()
      - OR a path to a .ckpt checkpoint / directory containing weights.

    Returns an eval()'d model on the requested device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if not os.path.exists(model_ref):
        try:
            m = NISQA()
            if hasattr(m, "from_pretrained"):
                model = m.from_pretrained(model_ref)
        except Exception:
            model = None

    if model is None:
        if not os.path.exists(model_ref):
            raise FileNotFoundError(
                f"Model reference not found and not loadable by from_pretrained(): {model_ref}"
            )
        if hasattr(NISQA, "load_from_checkpoint"):
            model = NISQA.load_from_checkpoint(model_ref, map_location=device)
        else:
            try:
                model = NISQA(model_ref)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load NISQA model. If you're using Lightning, pass the .ckpt path. "
                    "If you're using a model zoo build, pass the known model name."
                ) from e

    model.eval()
    if hasattr(model, "to"):
        model.to(device)
    return model

def nisqa_file(model, wav_path: str) -> float:
    """Predict MOS for a single WAV using the provided model."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    with torch.no_grad():
        out = model.predict_file(wav_path)
        if hasattr(out, "get") and "mos_pred" in out:
            val = out["mos_pred"]
            if hasattr(val, "values"):
                return float(val.values[0])
            return float(val)
        if isinstance(out, dict) and "mos_pred" in out:
            return float(out["mos_pred"])
        if isinstance(out, (list, tuple)) and len(out) > 0:
            return float(out[0])
        return float(out)

def nisqa_file_safe(model, wav_path: str, default: float = 2.5) -> float:
    """
    Safe NISQA. Returns finite MOS clamped to [1,5] or default on failures.
    """
    try:
        val = float(nisqa_file(model, wav_path))
        val = clamp(finite_or_default(val, default, "NISQA"), 1.0, 5.0)
        return val
    except Exception as e:
        LOG.warning("NISQA failed for %s: %s. Using default=%s.", wav_path, e, default)
        return float(default)
