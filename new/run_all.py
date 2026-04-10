#!/usr/bin/env python3
"""
Master script: runs the entire experimental pipeline.

Order:
  1. Prepare data (small subsets)
  2. For each task × seed:
     a. Train Base-clean
     b. Train Base-aug
     c. Calibrate gamma from Base-aug
     d. Train Plugin (with calibrated gamma)
  3. Evaluate all models on clean + every perturbation
  4. Aggregate results into results.csv
  5. Generate plots + summary.md
"""
import os, sys, json, csv, time, traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_cfg, RESULTS_DIR, CHECKPOINT_DIR, TASK_CFGS
from train import train
from gamma_calibrate import calibrate_gamma
from evaluate import evaluate_all
from data_utils import load_task_data
from analyze import generate_all


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(task_name: str, seed: int, device: str = "cuda:0",
                   alpha: float = 0.1, beta: float = 0.05):
    """Run full pipeline for one (task, seed) pair."""
    cfg_base = get_cfg(task_name, device=device)

    results_rows = []

    # ── 1. Train Base-clean ──────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"[1/4] Training Base-clean: {task_name}, seed={seed}")
    print(f"{'━'*70}")
    try:
        ckpt_clean = train(cfg_base, mode="base_clean", seed=seed)
    except Exception as e:
        print(f"  ⚠ Base-clean failed: {e}")
        traceback.print_exc()
        ckpt_clean = None

    # ── 2. Train Base-aug ────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"[2/4] Training Base-aug: {task_name}, seed={seed}")
    print(f"{'━'*70}")
    try:
        ckpt_aug = train(cfg_base, mode="base_aug", seed=seed)
    except Exception as e:
        print(f"  ⚠ Base-aug failed: {e}")
        traceback.print_exc()
        ckpt_aug = None

    # ── 3. Calibrate gamma ───────────────────────────────────────────────
    gamma_values = {}
    gamma_for_training = 0.0
    if ckpt_aug and os.path.isdir(ckpt_aug):
        print(f"\n{'━'*70}")
        print(f"[3/4] Calibrating gamma: {task_name}, seed={seed}")
        print(f"{'━'*70}")
        try:
            gamma_values = calibrate_gamma(cfg_base, ckpt_aug, seed=seed,
                                           quantiles=cfg_base["gamma_quantiles"])
            q = cfg_base["default_gamma_quantile"]
            gamma_for_training = gamma_values.get(q, 0.0)
            print(f"  Using gamma={gamma_for_training:.4f} (q{int(q*100)})")
        except Exception as e:
            print(f"  ⚠ Gamma calibration failed: {e}")
            traceback.print_exc()

    # ── 4. Train Plugin ──────────────────────────────────────────────────
    cfg_plugin = get_cfg(task_name, device=device, alpha=alpha, beta=beta)
    print(f"\n{'━'*70}")
    print(f"[4/4] Training Plugin: {task_name}, seed={seed}, gamma={gamma_for_training:.4f}")
    print(f"{'━'*70}")
    try:
        ckpt_plugin = train(cfg_plugin, mode="plugin",
                            gamma=gamma_for_training, seed=seed)
    except Exception as e:
        print(f"  ⚠ Plugin failed: {e}")
        traceback.print_exc()
        ckpt_plugin = None

    # ── 5. Evaluate all checkpoints ──────────────────────────────────────
    eval_gamma = gamma_for_training if gamma_for_training != 0.0 else 0.1

    confusion_data = []
    for mode, ckpt in [("base_clean", ckpt_clean),
                       ("base_aug", ckpt_aug),
                       ("plugin", ckpt_plugin)]:
        if ckpt is None or not os.path.isdir(ckpt):
            continue
        print(f"\nEvaluating {mode} ({ckpt})")
        cfg_eval = get_cfg(task_name, device=device, alpha=alpha, beta=beta)
        try:
            eval_results = evaluate_all(cfg_eval, ckpt,
                                        gamma=eval_gamma, seed=seed)
        except Exception as e:
            print(f"  ⚠ Evaluation failed for {mode}: {e}")
            traceback.print_exc()
            continue

        for r in eval_results:
            row = dict(
                task=task_name,
                modality=cfg_eval["modality"],
                perturbation=r["perturbation"],
                method=mode,
                seed=seed,
                accuracy=r["accuracy"],
                worst_class_acc=r["worst_class_acc"],
                worst_class_err=r["worst_class_err"],
                vwr_gamma=r["vwr_gamma"],
                sigma_max=r["sigma_max"],
                gamma=eval_gamma,
                alpha=alpha if mode == "plugin" else 0.0,
                beta=beta if mode == "plugin" else 0.0,
                kappa=cfg_eval["kappa"],
            )
            results_rows.append(row)

            if task_name == "robot" and r["perturbation"] != "clean":
                confusion_data.append(dict(
                    task=task_name, method=mode,
                    perturbation=r["perturbation"],
                    confusion=r["confusion"],
                    label_names=cfg_eval["label_names"],
                ))

    return results_rows, confusion_data, gamma_values


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["agnews", "arc", "robot"],
                        help="Tasks to run. Add 'scienceqa' for multimodal reasoning.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()

    all_rows = []
    all_confusion = []
    all_gamma = {}

    # Use task-specific CSV to avoid parallel overwrites
    task_label = "_".join(args.tasks)
    csv_path = os.path.join(RESULTS_DIR, f"results_{task_label}.csv")

    for task in args.tasks:
        for seed in args.seeds:
            t0 = time.time()
            try:
                rows, conf, gamma_vals = run_experiment(
                    task, seed, device=args.device,
                    alpha=args.alpha, beta=args.beta)
                all_rows.extend(rows)
                all_confusion.extend(conf)
                all_gamma[f"{task}_s{seed}"] = gamma_vals
            except Exception as e:
                print(f"\n{'!'*70}")
                print(f"FAILED: task={task} seed={seed}: {e}")
                traceback.print_exc()
                print(f"{'!'*70}")
            elapsed = time.time() - t0
            print(f"\n  [{task}/seed={seed}] completed in {elapsed/60:.1f} min")

            # Incremental save
            if all_rows:
                _write_csv(csv_path, all_rows)

    # Final save
    if all_rows:
        _write_csv(csv_path, all_rows)
        print(f"\nResults saved to {csv_path} ({len(all_rows)} rows)")

    # Save gamma values
    gamma_path = os.path.join(RESULTS_DIR, "gamma_values.json")
    with open(gamma_path, "w") as f:
        json.dump(all_gamma, f, indent=2)

    # Generate analysis
    print("\nGenerating analysis...")
    try:
        generate_all(csv_path, all_confusion)
    except Exception as e:
        print(f"Analysis generation failed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
