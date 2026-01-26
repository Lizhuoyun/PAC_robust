from src.eval.eval_classification import eval_classification


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    eval_classification(args.config, args.ckpt, overrides=list(args.override))
