import importlib
from typing import Dict, Any

def build_model_from_cfg(cfg: Dict[str, Any]):
    mcfg = cfg["model"]
    class_path = mcfg.get("class_path", "se_models.robustnet_plus.model:RobustTCNPlusSE")
    mod_name, cls_name = class_path.split(":")
    Mod = importlib.import_module(mod_name)
    Cls = getattr(Mod, cls_name)
    kwargs = {k:v for k,v in mcfg.items() if k not in ["class_path"]}
    return Cls(**kwargs)
