import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.datasets import get_dataset
from src.data.prompts import PromptRenderer, PromptTemplate, extract_mc_fields
from src.data.perturbations import build_perturbed_fields_cache, load_perturbed_fields_cache, materialize_perturbed_fields
from src.models.verbalizer import Verbalizer, margin_from_logits, resolve_verbalizer_ids
from src.logging.logger import ExperimentLogger
from src.train.utils import apply_overrides, load_config, resolve_presets, resolve_torch_dtype, save_json, set_seed


def _transition_matrix(preds: List[int], labels: List[int], num_classes: int) -> np.ndarray:
    """Eq.(5): empirical transition matrix; diagonal set to 0."""
    mat = np.zeros((num_classes, num_classes), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for p, y in zip(preds, labels):
        counts[y] += 1.0
        if p != y:
            mat[p, y] += 1.0
    counts[counts == 0] = 1.0
    mat = mat / counts
    np.fill_diagonal(mat, 0.0)
    return mat


def _transition_matrix_margin(
    preds: List[int],
    labels: List[int],
    margins: List[float],
    num_classes: int,
    gamma: float,
    tau: float,
) -> np.ndarray:
    """Eq.(7): margin-aware empirical transition matrix with g_gamma weights."""
    mat = np.zeros((num_classes, num_classes), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for p, y, m in zip(preds, labels, margins):
        counts[y] += 1.0
        if p != y:
            gate = 1.0 / (1.0 + np.exp(-(gamma - m) / tau))
            mat[p, y] += gate
    counts[counts == 0] = 1.0
    mat = mat / counts
    np.fill_diagonal(mat, 0.0)
    return mat


def wcr(mat: np.ndarray) -> float:
    """Eq.(6): worst-class robust risk as max column sum."""
    return float(mat.sum(axis=0).max())


def _load_model_and_tokenizer(cfg: Dict, ckpt: str, device: torch.device):
    # Tokenizer: prefer ckpt-local tokenizer if available.
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    torch_dtype = resolve_torch_dtype(cfg)
    if os.path.exists(adapter_cfg):
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
        model = PeftModel.from_pretrained(base, ckpt).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model, tokenizer


def eval_classification(config_path: str, ckpt: str, overrides: Optional[List[str]] = None) -> None:
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_presets(cfg)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    template = PromptTemplate(answer_marker=cfg["dataset"].get("answer_marker", "Answer: "))
    data = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["eval_split"])
    max_eval = cfg.get("dataset", {}).get("max_eval_examples") or cfg.get("dataset", {}).get("max_examples")
    if max_eval:
        data = data[: int(max_eval)]
    pert_cfg = cfg["perturbation"]

    labels = list(cfg["verbalizer"].keys())
    model, tokenizer = _load_model_and_tokenizer(cfg, ckpt, device)
    resolved_map, verbalizer_ids, tmpl = resolve_verbalizer_ids(
        tokenizer,
        labels=labels,
        marker=template.answer_marker,
        templates=cfg["verbalizer"].get("templates", None) if isinstance(cfg["verbalizer"], dict) else None,
    )
    verbalizer = Verbalizer(resolved_map)
    label_map = {label: idx for idx, label in enumerate(labels)}
    renderer = PromptRenderer(template, labels)
    # Build eval fields; skip examples whose gold label is not covered by the current verbalizer.
    # (ARC occasionally contains 5-choice items; if the config verbalizer is A-D, such examples
    # would otherwise crash evaluation with KeyError.)
    clean_fields_all = [extract_mc_fields(ex, labels) for ex in data]
    skipped = 0
    clean_fields: List[Dict] = []
    for f in clean_fields_all:
        if f.get("label") in label_map:
            clean_fields.append(f)
        else:
            skipped += 1
    if skipped > 0:
        print(f"[warn] skipped {skipped}/{len(clean_fields_all)} eval examples with label not in verbalizer: {sorted(set([f.get('label') for f in clean_fields_all if f.get('label') not in label_map]))}")
    metrics_logger = ExperimentLogger(cfg, cfg["logging"]["metrics_path"])

    def forward_logits(fields: Dict) -> torch.Tensor:
        prompt = renderer.render(fields)
        batch = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**batch).logits
        prompt_len = batch["attention_mask"].sum(dim=1) - 1
        pos_logits = logits[0, prompt_len[0]]
        return pos_logits.index_select(dim=-1, index=verbalizer_ids.to(device))

    clean_correct = 0
    for fields in clean_fields:
        label = label_map[fields["label"]]
        with torch.no_grad():
            clean_logits = forward_logits(fields)
            pred = int(clean_logits.argmax().item())
            if pred == label:
                clean_correct += 1

    budgets = cfg["perturbation"].get("eval_budgets") or [cfg["perturbation"].get("budget", None)]
    denom = max(len(clean_fields), 1)
    clean_metrics = {
        "clean_acc": clean_correct / denom,
        "n_eval": len(clean_fields),
        "n_skipped": skipped,
        "verbalizer_template": tmpl,
    }
    save_json(os.path.join(cfg["logging"]["save_dir"], "eval_clean.json"), clean_metrics)
    metrics = dict(clean_metrics)
    curves = []
    for budget in budgets:
        eval_cfg = dict(cfg["perturbation"])
        if budget is not None:
            eval_cfg["budget"] = budget
        cache_path = build_perturbed_fields_cache(
            clean_fields,
            dataset_name=cfg["dataset"]["name"],
            split=cfg["dataset"]["eval_split"],
            config=eval_cfg,
            seed=cfg.get("seed", 42),
            n_variants=1,
            cache_root=cfg["perturbation"].get("cache_root", "cache/perturb_fields"),
        )
        cache_records = load_perturbed_fields_cache(cache_path)
        pert_fields = materialize_perturbed_fields(clean_fields, cache_records)
        robust_correct = 0
        preds = []
        labels_idx = []
        margins = []
        for fields, pert in zip(clean_fields, pert_fields):
            label = label_map[fields["label"]]
            with torch.no_grad():
                pert_logits = forward_logits(pert)
                pred_p = int(pert_logits.argmax().item())
                if pred_p == label:
                    robust_correct += 1
                m = margin_from_logits(pert_logits.unsqueeze(0), torch.tensor([label], device=device))
            preds.append(pred_p)
            labels_idx.append(label)
            margins.append(float(m.item()))

        mat = _transition_matrix(preds, labels_idx, len(labels))
        mat_gamma = _transition_matrix_margin(preds, labels_idx, margins, len(labels), cfg["spectral"]["gamma"], cfg["spectral"]["tau"])
        sigma = float(torch.linalg.svdvals(torch.tensor(mat_gamma)).max().item())
        tag = "default" if budget is None else str(budget)
        metrics[f"robust_acc_{tag}"] = robust_correct / denom
        metrics[f"wcr_{tag}"] = wcr(mat_gamma)
        metrics[f"wcr_plain_{tag}"] = wcr(mat)
        metrics[f"sigma_max_{tag}"] = sigma
        curves.append({"budget": tag, "robust_acc": metrics[f"robust_acc_{tag}"], "wcr": metrics[f"wcr_{tag}"]})
        save_json(os.path.join(cfg["logging"]["save_dir"], f"matrix_{tag}.json"), {"matrix": mat.tolist(), "matrix_gamma": mat_gamma.tolist()})
        np.save(os.path.join(cfg["logging"]["save_dir"], f"matrix_{tag}.npy"), mat)
        np.save(os.path.join(cfg["logging"]["save_dir"], f"matrix_gamma_{tag}.npy"), mat_gamma)
        if tag == "default":
            save_json(cfg["logging"]["matrix_path"], {"matrix": mat.tolist(), "matrix_gamma": mat_gamma.tolist()})

    save_json(cfg["logging"]["final_metrics_path"], metrics)
    save_json(os.path.join(cfg["logging"]["save_dir"], "eval_robust.json"), metrics)
    if curves:
        csv_path = os.path.join(cfg["logging"]["save_dir"], "curves_budget.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("budget,robust_acc,wcr\n")
            for row in curves:
                f.write(f"{row['budget']},{row['robust_acc']:.6f},{row['wcr']:.6f}\n")
    metrics_logger.log({"event": "eval_complete", **metrics})
    metrics_logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    eval_classification(args.config, args.ckpt)
