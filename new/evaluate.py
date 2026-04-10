"""
Evaluation: clean accuracy, robust accuracy per perturbation,
worst-class metrics, VWR_gamma, sigma_max, confusion matrices.
"""
import os, sys, json, random, math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import get_cfg
from model_utils import (load_model_and_tokenizer, load_model_from_checkpoint,
                         get_label_token_ids,
                         tokenize_for_classification, extract_class_logits)
from data_utils import load_task_data, _agnews_prompt, _arc_prompt, _sqa_prompt, _robot_prompt
from perturbations import apply_perturbation
from plugin import compute_vwr_gamma, compute_sigma_max
from train import _process_multimodal_single


def _rebuild_prompt(ex, perturbed_text, task):
    if task == "agnews":
        return _agnews_prompt(perturbed_text)
    elif task == "arc":
        return _arc_prompt(perturbed_text, ex.get("choices", ["", "", "", ""]))
    elif task == "scienceqa":
        return _sqa_prompt(perturbed_text, ex.get("choices", []))
    elif task == "robot":
        from config import TASK_CFGS
        return _robot_prompt(perturbed_text, TASK_CFGS["robot"]["label_names"])
    return ex["prompt"]


@torch.no_grad()
def evaluate_split(model, tokenizer, processor, data: list, cfg: dict,
                   perturbation_type: str = "clean", gamma: float = 0.0,
                   seed: int = 42):
    """
    Evaluate on a data split with a specific perturbation.
    Returns dict of metrics.
    """
    device = next(model.parameters()).device
    label_token_ids = get_label_token_ids(tokenizer, cfg["label_chars"])
    num_classes = cfg["num_classes"]
    is_mm = cfg["modality"] == "multimodal"
    bs = cfg["batch_size"] if not is_mm else 1
    rng = random.Random(seed)

    all_preds, all_labels, all_logits = [], [], []

    for start in tqdm(range(0, len(data), bs), desc=f"Eval {perturbation_type}", leave=False):
        batch = data[start:start + bs]

        if perturbation_type != "clean":
            proc_batch = []
            for ex in batch:
                ptxt, pimg = apply_perturbation(
                    ex["text"], ex.get("image"), perturbation_type,
                    ex["task"], ex.get("fill_fields"), cfg, rng)
                ex_copy = dict(ex)
                ex_copy["text"] = ptxt
                ex_copy["prompt"] = _rebuild_prompt(ex, ptxt, ex["task"])
                if pimg is not None:
                    ex_copy["image"] = pimg
                proc_batch.append(ex_copy)
            batch = proc_batch

        if is_mm:
            for ex in batch:
                inp, lp = _process_multimodal_single(
                    processor, ex["prompt"], ex["label_char"],
                    ex.get("image"), device, cfg["max_seq_len"])
                out = model(**inp)
                cl = extract_class_logits(out.logits,
                                          torch.tensor([lp], device=device),
                                          label_token_ids)
                all_logits.append(cl.cpu())
                all_preds.append(cl.argmax(dim=-1).cpu())
                all_labels.append(torch.tensor([ex["label"]]))
        else:
            prompts = [e["prompt"] for e in batch]
            lcs = [e["label_char"] for e in batch]
            labs = torch.tensor([e["label"] for e in batch])
            enc = tokenize_for_classification(tokenizer, prompts, lcs, cfg["max_seq_len"])
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cl = extract_class_logits(out.logits, enc["label_positions"], label_token_ids)
            all_logits.append(cl.cpu())
            all_preds.append(cl.argmax(dim=-1).cpu())
            all_labels.append(labs)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits_all = torch.cat(all_logits, dim=0)

    # Overall accuracy
    correct = (preds == labels).float()
    acc = correct.mean().item()

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[c] = correct[mask].mean().item()
        else:
            per_class_acc[c] = float("nan")

    worst_class_acc = min(v for v in per_class_acc.values() if not math.isnan(v))
    worst_class_err = 1.0 - worst_class_acc

    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, l in zip(preds, labels):
        confusion[p.item(), l.item()] += 1

    # VWR_gamma and sigma_max
    kappa = cfg.get("kappa", 0.5)
    vwr = compute_vwr_gamma(logits_all, labels, gamma, kappa, num_classes)
    smax = compute_sigma_max(logits_all, labels, gamma, kappa, num_classes)

    return dict(
        accuracy=acc,
        per_class_acc=per_class_acc,
        worst_class_acc=worst_class_acc,
        worst_class_err=worst_class_err,
        vwr_gamma=vwr,
        sigma_max=smax,
        confusion=confusion.tolist(),
        n_samples=len(labels),
    )


def evaluate_all(cfg: dict, checkpoint_dir: str, gamma: float = 0.0, seed: int = 42):
    """
    Evaluate a trained model on clean + all perturbation types.
    Returns list of result dicts.
    """
    model, tokenizer, processor = load_model_from_checkpoint(cfg, checkpoint_dir)
    model.eval()
    data = load_task_data(cfg, seed=seed)
    test_data = data["test"]

    perturbation_types = ["clean"] + cfg["perturbation_types"]
    results = []

    for ptype in perturbation_types:
        print(f"  Evaluating perturbation: {ptype}")
        metrics = evaluate_split(model, tokenizer, processor, test_data, cfg,
                                 perturbation_type=ptype, gamma=gamma, seed=seed)
        metrics["perturbation"] = ptype
        results.append(metrics)
        print(f"    Acc={metrics['accuracy']:.4f}  Worst={metrics['worst_class_acc']:.4f}"
              f"  VWR={metrics['vwr_gamma']:.4f}  σ_max={metrics['sigma_max']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_model_from_checkpoint(task: str, mode: str, seed: int,
                                    gamma: float = 0.0, device: str = "cuda:0",
                                    alpha: float = None, beta: float = None):
    """
    Load a saved checkpoint and evaluate.  Returns list of result dicts.
    """
    overrides = dict(device=device)
    if alpha is not None:
        overrides["alpha"] = alpha
    if beta is not None:
        overrides["beta"] = beta
    cfg = get_cfg(task, **overrides)

    ckpt_dir = os.path.join(cfg["checkpoint_dir"],
                            f"{task}_{mode}_s{seed}")
    if not os.path.isdir(ckpt_dir):
        print(f"  [SKIP] Checkpoint not found: {ckpt_dir}")
        return []

    return evaluate_all(cfg, ckpt_dir, gamma=gamma, seed=seed)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    results = evaluate_model_from_checkpoint(args.task, args.mode, args.seed,
                                             gamma=args.gamma, device=args.device)
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "confusion"}, indent=2))
