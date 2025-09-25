def main(cfg):
    from se_cli.utils.seed import set_random_seeds
    set_random_seeds(int(cfg.get("seed", 1337)))
    print("[train] cfg:", cfg)
