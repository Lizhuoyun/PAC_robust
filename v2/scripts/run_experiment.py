#!/usr/bin/env python3
"""
Single experiment runner: one (model, task, method, seed, gamma_q) combination.

Usage:
  python scripts/run_experiment.py --method r3f_plugin --seed 42 --gamma_q 0.25
  python scripts/run_experiment.py --method base_aug --seed 42 --task boolq --model_path /path/to/model
"""
import sys, os, argparse, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config          import get_cfg
from data            import load_data
from models          import load_from_checkpoint
from trainers        import make_trainer
from train_engine    import run_training
from eval            import evaluate_all
from gamma_calibrate import calibrate_gamma


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",     required=True)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--gamma_q",    type=float, default=0.25)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--task",       default=None, help="arc or boolq")
    p.add_argument("--model_path", default=None, help="Override model path")
    p.add_argument("--model_tag",  default=None, help="Short name for model (e.g. qwen05b)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--alpha",      type=float, default=None)
    p.add_argument("--beta",       type=float, default=None)
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()


def _needs_gamma(method):
    return method.endswith("_plugin") or method == "plugin"


def _fmt_tag_float(x: float) -> str:
    s = f"{x:.4g}"
    return s.replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    overrides = dict(device=args.device)
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.beta is not None:
        overrides["beta"] = args.beta
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum
    if args.model_path is not None:
        overrides["model_name"] = args.model_path
    if args.task is not None:
        overrides["task_name"] = args.task
    cfg = get_cfg(**overrides)

    # Derive per-model-task directories
    model_tag = args.model_tag or os.path.basename(cfg["model_name"].rstrip("/"))
    task_tag  = cfg["task_name"]
    results_dir = os.path.join(cfg["results_dir"], f"{model_tag}_{task_tag}")
    ckpt_dir_root = os.path.join(cfg["ckpt_dir"], f"{model_tag}_{task_tag}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir_root, exist_ok=True)

    # Build unique tag
    if _needs_gamma(args.method):
        tag = f"{args.method}_q{int(args.gamma_q*100):02d}_s{args.seed}"
    else:
        tag = f"{args.method}_s{args.seed}"
    if args.alpha is not None:
        tag += f"_a{_fmt_tag_float(args.alpha)}"
    if args.beta is not None:
        tag += f"_b{_fmt_tag_float(args.beta)}"

    # ── Determine gamma ────────────────────────────────────────────────
    gamma = 0.0
    if _needs_gamma(args.method):
        aug_ckpt = os.path.join(ckpt_dir_root, f"base_aug_s{args.seed}")
        if not os.path.isdir(aug_ckpt):
            raise FileNotFoundError(
                f"Base-aug checkpoint not found: {aug_ckpt}\n"
                f"Run base_aug first.")

        gamma_cache = os.path.join(results_dir, f"gamma_s{args.seed}.json")
        lock_path   = gamma_cache + ".lock"
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
    ckpt_path = os.path.join(ckpt_dir_root, tag)
    # Override CKPT_DIR for train_engine
    cfg["_ckpt_dir_root"] = ckpt_dir_root

    if os.path.isdir(ckpt_path) and not args.force_retrain:
        print(f"\n  [SKIP train] Checkpoint exists: {ckpt_path}")
    else:
        trainer = make_trainer(
            args.method,
            gamma=gamma if _needs_gamma(args.method) else None,
            cfg=cfg)
        # Monkey-patch CKPT_DIR for this run
        import config as _cfg_mod
        _orig_ckpt = _cfg_mod.CKPT_DIR
        _cfg_mod.CKPT_DIR = ckpt_dir_root
        try:
            ckpt_path = run_training(trainer, cfg, seed=args.seed, tag=tag)
        finally:
            _cfg_mod.CKPT_DIR = _orig_ckpt

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n── Evaluating {tag} ──")
    model, tok = load_from_checkpoint(cfg, ckpt_path)
    model.eval()
    data = load_data(cfg, seed=args.seed)

    eval_gamma = gamma if gamma != 0.0 else 0.1
    results = evaluate_all(model, tok, data["test"], cfg,
                           gamma=eval_gamma, seed=args.seed)
    del model
    torch.cuda.empty_cache()

    # ── Save results ───────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append(dict(
            model_tag            = model_tag,
            task                 = task_tag,
            method               = args.method,
            seed                 = args.seed,
            alpha                = cfg["alpha"],
            beta                 = cfg["beta"],
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

    csv_path = os.path.join(results_dir, f"results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {csv_path}  ({len(rows)} rows)")

    # Free disk: delete checkpoint after eval unless it's base_aug
    # (base_aug is needed for gamma calibration by plugin jobs)
    if args.method != "base_aug" and os.path.isdir(ckpt_path):
        import shutil
        shutil.rmtree(ckpt_path, ignore_errors=True)
        print(f"  [CLEANUP] Removed checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
