from src.eval.eval_generation import eval_generation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    eval_generation(args.config, args.ckpt, overrides=list(args.override))
