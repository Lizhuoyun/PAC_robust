#!/usr/bin/env python3
"""
Master orchestration script.

Runs all Priority-1 methods in the correct order for all seeds:
  1. base_clean   (no dependency)
  2. base_aug     (no dependency)
  3. Gamma calibration (requires base_aug)
  4. plugin       (requires gamma)
  5. r3f          (no dependency)
  6. r3f_plugin   (requires gamma)
  7. smart        (no dependency)
  8. smart_plugin (requires gamma)
  Optional P2:
  9. awp, awp_plugin

Usage:
  python scripts/run_all.py [--seeds 42 123] [--p2] [--device cuda:0]
  python scripts/run_all.py --skip_trained  # skip existing checkpoints
"""
import sys, os, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEEDS, GAMMA_QUANTILES, DEFAULT_GAMMA_Q, CKPT_DIR, RESULTS_DIR

PYTHON = sys.executable

def run(cmd: list, desc: str):
    print(f"\n{'─'*64}")
    print(f"  ▶  {desc}")
    print(f"{'─'*64}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"  [WARN] Non-zero exit code {ret.returncode} for: {desc}")
    return ret.returncode


def experiment_cmd(method: str, seed: int, gamma_q: float,
                   device: str, skip: bool) -> list:
    cmd = [PYTHON, "scripts/run_experiment.py",
           "--method",  method,
           "--seed",    str(seed),
           "--gamma_q", str(gamma_q),
           "--device",  device]
    if skip:
        cmd.append("--force_retrain" if not skip else "")
    return [c for c in cmd if c]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",  nargs="+", type=int, default=None)
    p.add_argument("--p2",     action="store_true",
                   help="Also run P2 methods (AWP)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gamma_q", type=float, default=DEFAULT_GAMMA_Q,
                   help="Default gamma quantile for plugin methods")
    p.add_argument("--skip_trained", action="store_true",
                   help="Skip if checkpoint already exists")
    p.add_argument("--no_base", action="store_true",
                   help="Skip base_clean and base_aug (if already trained)")
    return p.parse_args()


def ckpt_exists(method: str, seed: int, gamma_q: float) -> bool:
    from scripts.run_experiment import _ckpt_tag
    tag = _ckpt_tag(method, seed, gamma_q)
    return os.path.isdir(os.path.join(CKPT_DIR, tag))


def main():
    args  = parse_args()
    seeds = args.seeds or SEEDS
    q     = args.gamma_q
    dev   = args.device

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    # Change to v2 root so relative paths work
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    for seed in seeds:
        print(f"\n{'═'*64}")
        print(f"  SEED = {seed}")
        print(f"{'═'*64}")

        # ── P1 methods ─────────────────────────────────────────────────
        if not args.no_base:
            run([PYTHON, "scripts/run_experiment.py",
                 "--method", "base_clean", "--seed", str(seed),
                 "--device", dev],
                f"base_clean / seed={seed}")

            run([PYTHON, "scripts/run_experiment.py",
                 "--method", "base_aug", "--seed", str(seed),
                 "--device", dev],
                f"base_aug / seed={seed}")

        run([PYTHON, "scripts/run_experiment.py",
             "--method", "plugin", "--seed", str(seed),
             "--gamma_q", str(q), "--device", dev],
            f"plugin (q={q}) / seed={seed}")

        run([PYTHON, "scripts/run_experiment.py",
             "--method", "r3f", "--seed", str(seed), "--device", dev],
            f"r3f / seed={seed}")

        run([PYTHON, "scripts/run_experiment.py",
             "--method", "r3f_plugin", "--seed", str(seed),
             "--gamma_q", str(q), "--device", dev],
            f"r3f_plugin (q={q}) / seed={seed}")

        run([PYTHON, "scripts/run_experiment.py",
             "--method", "smart", "--seed", str(seed), "--device", dev],
            f"smart / seed={seed}")

        run([PYTHON, "scripts/run_experiment.py",
             "--method", "smart_plugin", "--seed", str(seed),
             "--gamma_q", str(q), "--device", dev],
            f"smart_plugin (q={q}) / seed={seed}")

        # ── P2 methods ─────────────────────────────────────────────────
        if args.p2:
            run([PYTHON, "scripts/run_experiment.py",
                 "--method", "awp", "--seed", str(seed), "--device", dev],
                f"awp / seed={seed}")

            run([PYTHON, "scripts/run_experiment.py",
                 "--method", "awp_plugin", "--seed", str(seed),
                 "--gamma_q", str(q), "--device", dev],
                f"awp_plugin (q={q}) / seed={seed}")

    # ── Aggregate after all seeds ──────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  Aggregating results …")
    print(f"{'═'*64}")
    run([PYTHON, "scripts/aggregate.py", "--device", dev],
        "aggregate results")

    print("\n  ✓ run_all.py complete.")


if __name__ == "__main__":
    main()
