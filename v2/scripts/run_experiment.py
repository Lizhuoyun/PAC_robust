#!/usr/bin/env python3
"""
Single experiment runner: one (method, seed, gamma_quantile) combination.

Usage:
  python scripts/run_experiment.py --method r3f_plugin --seed 42 --gamma_q 0.25

  method options:
    base_clean, base_aug, plugin,
    r3f, r3f_plugin,
    smart, smart_plugin,
    awp, awp_plugin
"""
import sys, os, argparse, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config          import get_cfg, RESULTS_DIR, CKPT_DIR
from data            import load_arc
from models          import load_from_checkpoint
from trainers        import make_trainer
from train_engine    import run_training
from eval            import evaluate_all
from gamma_calibrate import calibrate_gamma


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",   required=True)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--gamma_q",  type=float, default=0.25)
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--alpha",    type=float, default=None)
    p.add_argument("--beta",     type=float, default=None)
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()


def _ckpt_tag(method: str, seed: int, gamma_q: float = None) -> str:
    if method.endswith("_plugin") or method == "plugin":
        return f"{method}_q{int(gamma_q*100):02d}_s{seed}"
    return f"{method}_s{seed}"


def _needs_gamma(method: str) -> bool:
    return method.endswith("_plugin") or method == "plugin"


def main():
    args = parse_args()
    overrides = dict(device=args.device)
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.beta is not None:
        overrides["beta"] = args.beta
    cfg = get_cfg(**overrides)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    # ── Determine gamma ────────────────────────────────────────────────
    gamma = 0.0
    if _needs_gamma(args.method):
        aug_ckpt = os.path.join(CKPT_DIR, f"base_aug_s{args.seed}")
        if not os.path.isdir(aug_ckpt):
            raise FileNotFoundError(
                f"Base-aug checkpoint not found: {aug_ckpt}\n"
                f"Run base_aug first: python scripts/run_experiment.py "
                f"--method base_aug --seed {args.seed}")

        gamma_cache = os.path.join(RESULTS_DIR, f"gamma_s{args.seed}.json")
        lock_path   = gamma_cache + ".lock"
        # File lock prevents parallel plugin jobs from calibrating simultaneously
        import fcntl
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            if os.path.exists(gamma_cache):
                gamma_vals = {float(k): float(v)
                              for k, v in json.load(open(gamma_cache)).items()}
            else:
                print(f"\n── Calibrating gamma (seed={args.seed}) ──")
                gamma_vals = calibrate_gamma(cfg, aug_ckpt, seed=args.seed)
                with open(gamma_cache, "w") as f:
                    json.dump({str(k): v for k, v in gamma_vals.items()},
                              f, indent=2)
            fcntl.flock(lock_f, fcntl.LOCK_UN)

        gamma = gamma_vals.get(args.gamma_q)
        if gamma is None:
            raise KeyError(f"gamma_q={args.gamma_q} not found in {gamma_vals}")
        print(f"  gamma = {gamma:.4f}  (q{int(args.gamma_q*100):02d})")

    # ── Train ──────────────────────────────────────────────────────────
    tag      = _ckpt_tag(args.method, args.seed, args.gamma_q)
    ckpt_dir = os.path.join(CKPT_DIR, tag)

    if os.path.isdir(ckpt_dir) and not args.force_retrain:
        print(f"\n  [SKIP train] Checkpoint exists: {ckpt_dir}")
    else:
        trainer  = make_trainer(
            args.method,
            gamma = gamma if _needs_gamma(args.method) else None,
            cfg   = cfg)
        ckpt_dir = run_training(trainer, cfg, seed=args.seed, tag=tag)

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n── Evaluating {tag} ──")
    model, tok = load_from_checkpoint(cfg, ckpt_dir)
    model.eval()
    data       = load_arc(cfg, seed=args.seed)

    eval_gamma = gamma if gamma != 0.0 else 0.1
    results    = evaluate_all(model, tok, data["test"], cfg,
                              gamma=eval_gamma, seed=args.seed)
    del model
    torch.cuda.empty_cache()

    # ── Save results ───────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append(dict(
            method               = args.method,
            seed                 = args.seed,
            gamma_q              = args.gamma_q,
            gamma                = eval_gamma,
            perturbation         = r["perturbation"],
            accuracy             = r["accuracy"],
            worst_class_acc      = r["worst_class_acc"],
            worst_class_err      = r["worst_class_err"],
            vwr_gamma            = r["vwr_gamma"],
            sigma_max            = r["sigma_max"],
            fragile_ratio        = r.get("fragile_ratio", float("nan")),
            mean_gate            = r.get("mean_gate",    float("nan")),
            clean_to_robust_drop = r.get("clean_to_robust_drop", 0.0),
        ))

    csv_path = os.path.join(RESULTS_DIR, f"results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
