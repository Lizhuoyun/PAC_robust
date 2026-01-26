import glob
import os
import subprocess

SUITES = {
    "table1": "configs/classification/arc/*.yaml",
    "table2": "configs/classification/csqa/*.yaml",
    "table3": "configs/generation/gsm8k/*.yaml",
    # Optional suites (may not exist in a minimal checkout)
    "ablations": "configs/ablations/*.yaml",
    "curves": "configs/curves/*.yaml",
    "efficiency": "configs/efficiency/*.yaml",
}


def run_suite(suite: str, overrides: list = None) -> None:
    pattern = SUITES.get(suite)
    if not pattern:
        raise ValueError(f"unknown suite {suite}")
    configs = glob.glob(pattern)
    if not configs:
        available = ", ".join(sorted(SUITES.keys()))
        raise ValueError(f"no configs matched suite={suite} (pattern={pattern}). Available suites: {available}")
    for cfg in configs:
        if "generation" in cfg:
            cmd = ["python", "-m", "experiments.train_generation", "--config", cfg]
        else:
            cmd = ["python", "-m", "experiments.train_classification", "--config", cfg]
        for ov in (overrides or []):
            cmd.extend(["--override", ov])
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run_suite(args.suite, overrides=list(args.override))
