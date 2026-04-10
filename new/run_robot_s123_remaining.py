#!/usr/bin/env python3
"""
Complete remaining Robot seed=123 experiment steps:
  - Gamma calibration from existing base_aug checkpoint
  - Plugin training
  - Evaluation of all 3 methods (base_clean, base_aug, plugin)
  - Save results to results_robot_s123.csv
"""
import os, sys, json, csv, traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_cfg, RESULTS_DIR, CHECKPOINT_DIR
from train import train
from gamma_calibrate import calibrate_gamma
from evaluate import evaluate_all

TASK = "robot"
SEED = 123
DEVICE = "cuda:0"
ALPHA = 0.1
BETA = 0.05

def main():
    cfg_base = get_cfg(TASK, device=DEVICE)

    ckpt_clean = os.path.join(CHECKPOINT_DIR, f"{TASK}_base_clean_s{SEED}")
    ckpt_aug = os.path.join(CHECKPOINT_DIR, f"{TASK}_base_aug_s{SEED}")

    assert os.path.isdir(ckpt_clean), f"Missing checkpoint: {ckpt_clean}"
    assert os.path.isdir(ckpt_aug), f"Missing checkpoint: {ckpt_aug}"
    print(f"Found existing checkpoints:\n  {ckpt_clean}\n  {ckpt_aug}")

    # 1. Gamma calibration
    print(f"\n{'━'*70}")
    print(f"[1/3] Calibrating gamma: {TASK}, seed={SEED}")
    print(f"{'━'*70}")
    gamma_values = calibrate_gamma(cfg_base, ckpt_aug, seed=SEED,
                                    quantiles=cfg_base["gamma_quantiles"])
    q = cfg_base["default_gamma_quantile"]
    gamma_for_training = gamma_values.get(q, 0.0)
    print(f"  Using gamma={gamma_for_training:.4f} (q{int(q*100)})")

    # 2. Train Plugin
    cfg_plugin = get_cfg(TASK, device=DEVICE, alpha=ALPHA, beta=BETA)
    print(f"\n{'━'*70}")
    print(f"[2/3] Training Plugin: {TASK}, seed={SEED}, gamma={gamma_for_training:.4f}")
    print(f"{'━'*70}")
    ckpt_plugin = train(cfg_plugin, mode="plugin",
                        gamma=gamma_for_training, seed=SEED)

    # 3. Evaluate all
    eval_gamma = gamma_for_training if gamma_for_training != 0.0 else 0.1
    results_rows = []

    for mode, ckpt in [("base_clean", ckpt_clean),
                       ("base_aug", ckpt_aug),
                       ("plugin", ckpt_plugin)]:
        if ckpt is None or not os.path.isdir(ckpt):
            print(f"  Skipping {mode}: no checkpoint")
            continue
        print(f"\n{'━'*70}")
        print(f"[3/3] Evaluating {mode} ({ckpt})")
        print(f"{'━'*70}")
        cfg_eval = get_cfg(TASK, device=DEVICE, alpha=ALPHA, beta=BETA)
        eval_results = evaluate_all(cfg_eval, ckpt, gamma=eval_gamma, seed=SEED)

        for r in eval_results:
            row = dict(
                task=TASK,
                modality=cfg_eval["modality"],
                perturbation=r["perturbation"],
                method=mode,
                seed=SEED,
                accuracy=r["accuracy"],
                worst_class_acc=r["worst_class_acc"],
                worst_class_err=r["worst_class_err"],
                vwr_gamma=r["vwr_gamma"],
                sigma_max=r["sigma_max"],
                gamma=eval_gamma,
                alpha=ALPHA if mode == "plugin" else 0.0,
                beta=BETA if mode == "plugin" else 0.0,
                kappa=cfg_eval["kappa"],
            )
            results_rows.append(row)

    # Save
    csv_path = os.path.join(RESULTS_DIR, "results_robot_s123.csv")
    if results_rows:
        fields = list(results_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results_rows)
        print(f"\nResults saved to {csv_path} ({len(results_rows)} rows)")

    gamma_path = os.path.join(RESULTS_DIR, "gamma_robot_s123.json")
    with open(gamma_path, "w") as f:
        json.dump(gamma_values, f, indent=2)

    print("\nDONE: Robot seed 123 complete!")

if __name__ == "__main__":
    main()
