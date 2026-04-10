#!/usr/bin/env python3
"""
Gamma sweep experiment.

For each (seed, gamma_quantile in [q10, q25, q50]):
  - Train plugin model
  - Evaluate
  - Save results with gamma_q label

Requires base_aug checkpoints + gamma calibration JSON.
Output: results/gamma_sweep_results.csv + gamma_sweep_plot_data.json

Usage:
  python scripts/gamma_sweep.py [--seeds 42 123] [--device cuda:0]
"""
import sys, os, subprocess, argparse, json, csv, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEEDS, GAMMA_QUANTILES, RESULTS_DIR, CKPT_DIR

PYTHON = sys.executable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",     nargs="+", type=int, default=None)
    p.add_argument("--methods",   nargs="+",
                   default=["plugin", "r3f_plugin", "smart_plugin"])
    p.add_argument("--device",    default="cuda:0")
    return p.parse_args()


def main():
    args   = parse_args()
    seeds  = args.seeds or SEEDS
    root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for seed in seeds:
        for q in GAMMA_QUANTILES:
            for method in args.methods:
                tag  = f"{method}_q{int(q*100):02d}_s{seed}"
                ckpt = os.path.join(CKPT_DIR, tag)
                skip = os.path.isdir(ckpt)
                cmd  = [PYTHON, "scripts/run_experiment.py",
                        "--method",   method,
                        "--seed",     str(seed),
                        "--gamma_q",  str(q),
                        "--device",   args.device]
                if skip:
                    print(f"  [SKIP]  {tag}")
                    continue
                print(f"\n── {tag} ──")
                ret = subprocess.run(cmd, check=False)
                if ret.returncode != 0:
                    print(f"  [WARN] {tag} exited {ret.returncode}")

    # ── Collect gamma sweep CSV ────────────────────────────────────────
    print("\n── Collecting gamma sweep results ──")
    all_rows = []
    for q in GAMMA_QUANTILES:
        for seed in seeds:
            for method in args.methods:
                tag = f"{method}_q{int(q*100):02d}_s{seed}"
                fp  = os.path.join(RESULTS_DIR, f"results_{tag}.csv")
                if not os.path.exists(fp):
                    print(f"  [MISS] {fp}")
                    continue
                with open(fp) as f:
                    for row in csv.DictReader(f):
                        all_rows.append(row)

    if not all_rows:
        print("  No results found for gamma sweep.")
        return

    out = os.path.join(RESULTS_DIR, "gamma_sweep_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"  Saved → {out}  ({len(all_rows)} rows)")

    # ── Build plot data JSON (for gamma sweep curves) ──────────────────
    # Structure: {method: {perturbation: {q: {metric: value}}}}
    from collections import defaultdict
    import statistics

    plot = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    metrics_to_report = ["accuracy", "worst_class_acc", "vwr_gamma",
                         "sigma_max", "clean_to_robust_drop"]

    # Group by (method, gamma_q, perturbation)
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["method"], float(row["gamma_q"]), row["perturbation"])
        grouped[key].append(row)

    for (method, gq, ptype), rows in grouped.items():
        for m in metrics_to_report:
            vals = [float(r[m]) for r in rows if r.get(m) not in ("", "nan", None)]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            plot[method][ptype][gq][m] = {"mean": mean, "std": std}

    json_out = os.path.join(RESULTS_DIR, "gamma_sweep_plot_data.json")
    with open(json_out, "w") as f:
        json.dump(plot, f, indent=2)
    print(f"  Saved → {json_out}")


if __name__ == "__main__":
    main()
