def main(cfg):
    from se_cli.utils.seed import set_random_seeds
    set_random_seeds(int(cfg.get("seed", 1337)))

    # build model (stub)
    import importlib
    model_name = cfg.get("model", {}).get("name", "mamba_unet")
    try:
        mod = importlib.import_module(f"se_models.{model_name}")
        model = mod.build_model(cfg)
    except Exception as e:
        print(f"[train] WARN: nu pot încărca modelul '{model_name}': {e}")
        model = None

    print("[train] cfg ok. model:", type(model).__name__ if model else None)

    # smoke loop minimal (fără date reale)
    if model:
        import torch
        x = torch.randn(2, 16000)  # 2 eșantioane, 1s @16k
        y = model(x)
        print("[train] forward ok:", tuple(y.shape))
