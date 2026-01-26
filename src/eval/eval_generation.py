import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.datasets import get_dataset
from src.data.prompts import PromptRenderer, PromptTemplate, extract_gen_fields
from src.data.perturbations import build_perturbed_fields_cache, load_perturbed_fields_cache, materialize_perturbed_fields
from src.losses.spectral import batch_transition_matrix
from src.logging.logger import ExperimentLogger
from src.train.utils import apply_overrides, load_config, resolve_presets, resolve_torch_dtype, save_json, set_seed


def _extract_final(text: str, marker: str) -> str:
    if marker in text:
        res = text.split(marker, 1)[1].strip().split("\n")[0]
    else:
        res = text.strip().split("\n")[0]
    res = res.rstrip(".!?;")
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", res)
    if nums:
        return nums[-1].replace(",", "")
    return res.strip()


def _candidate_probs(logits: torch.Tensor, gold_ids: torch.Tensor, top_k: int) -> torch.Tensor:
    masked = logits.clone()
    masked[torch.arange(logits.size(0), device=logits.device), gold_ids] = -1e9
    topk = torch.topk(masked, k=top_k, dim=-1)
    cand_ids = torch.cat([gold_ids.unsqueeze(1), topk.indices], dim=1)
    cand_logits = logits.gather(1, cand_ids)
    return F.softmax(cand_logits, dim=-1)


def _sample_positions(mask: torch.Tensor, sample_n: int) -> torch.Tensor:
    if sample_n <= 0:
        return mask
    flat_idx = mask.nonzero(as_tuple=False)
    if flat_idx.size(0) <= sample_n:
        return mask
    perm = torch.randperm(flat_idx.size(0), device=mask.device)[:sample_n]
    chosen = flat_idx[perm]
    out = torch.zeros_like(mask)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def _load_model_and_tokenizer(cfg: Dict, ckpt: str, device: torch.device):
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


def eval_generation(config_path: str, ckpt: str, overrides: Optional[List[str]] = None) -> None:
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_presets(cfg)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    template = PromptTemplate(final_marker=cfg["dataset"].get("final_marker", "Final answer: "))
    data = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["eval_split"])
    max_eval = cfg.get("dataset", {}).get("max_eval_examples") or cfg.get("dataset", {}).get("max_examples")
    if max_eval:
        data = data[: int(max_eval)]

    model, tokenizer = _load_model_and_tokenizer(cfg, ckpt, device)
    metrics_logger = ExperimentLogger(cfg, cfg["logging"]["metrics_path"])

    renderer = PromptRenderer(template)
    clean_fields = [extract_gen_fields(ex) for ex in data]
    clean_correct = 0
    for fields in clean_fields:
        prompt = renderer.render(fields)
        clean_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            clean_out = model.generate(**clean_ids, max_new_tokens=cfg["generation"]["max_new_tokens"])
        clean_text = tokenizer.decode(clean_out[0], skip_special_tokens=True)
        clean_ans = _extract_final(clean_text, template.final_marker)
        if clean_ans == fields["answer"]:
            clean_correct += 1

    budgets = cfg["perturbation"].get("eval_budgets") or [cfg["perturbation"].get("budget", None)]
    clean_metrics = {"clean_em": clean_correct / len(data)}
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
        for fields, pert in zip(clean_fields, pert_fields):
            pert_prompt = renderer.render(pert)
            pert_ids = tokenizer(pert_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                pert_out = model.generate(**pert_ids, max_new_tokens=cfg["generation"]["max_new_tokens"])
            pert_text = tokenizer.decode(pert_out[0], skip_special_tokens=True)
            pert_ans = _extract_final(pert_text, template.final_marker)
            if pert_ans == fields["answer"]:
                robust_correct += 1

        top_k = int(cfg["spectral"]["top_k"])
        num_classes = top_k + 1
        mat_num = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        counts = torch.zeros(num_classes, dtype=torch.float32)
        gate_sum = 0.0
        pos_count = 0
        for pert in pert_fields:
            prompt = renderer.render(pert)
            answer = pert["answer"]
            full = prompt + " " + answer
            tokens = tokenizer(full, return_tensors="pt").to(device)
            input_ids = tokens["input_ids"]
            labels = input_ids.clone()
            prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            labels[:, : prompt_ids.size(1)] = -100
            with torch.no_grad():
                logits = model(**tokens).logits
            mask = labels != -100
            if cfg["generation"].get("positions", "answer") == "uniform":
                mask = _sample_positions(mask, cfg["generation"].get("sample_n", 0))
            pos_indices = mask.nonzero(as_tuple=False)
            if pos_indices.numel() == 0:
                continue
            logits_pos = logits[pos_indices[:, 0], pos_indices[:, 1]]
            gold_ids = labels[pos_indices[:, 0], pos_indices[:, 1]]
            probs = _candidate_probs(logits_pos, gold_ids, top_k)
            gold_logit = logits_pos.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
            max_comp = logits_pos.topk(top_k, dim=-1).values.max(dim=1).values
            margins = gold_logit - max_comp
            gate = torch.sigmoid((cfg["spectral"]["gamma"] - margins) / cfg["spectral"]["tau"]).float()
            gate_sum += float(gate.sum().item())
            pos_count += int(gate.numel())
            labels_local = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)
            one_hot = F.one_hot(labels_local, num_classes=num_classes).to(dtype=torch.float32)
            weighted = probs.float() * gate.unsqueeze(1)
            mat_num += (weighted.t() @ one_hot).detach().cpu()
            counts += one_hot.sum(dim=0).detach().cpu()

        tag = "default" if budget is None else str(budget)
        if pos_count > 0:
            denom = counts.clamp_min(1.0)
            mat = mat_num / denom.unsqueeze(0)
            mat = mat * (1.0 - torch.eye(num_classes, dtype=mat.dtype))
            sigma = float(torch.linalg.svdvals(mat).max().item())
            token_risk = float(gate_sum / max(pos_count, 1))
            npy_path = os.path.join(cfg["logging"]["save_dir"], f"token_matrix_{tag}.npy")
            np.save(npy_path, mat.numpy())
        else:
            sigma = 0.0
            token_risk = 0.0

        metrics[f"robust_em_{tag}"] = robust_correct / len(data)
        metrics[f"token_risk_{tag}"] = token_risk
        metrics[f"sigma_max_{tag}"] = sigma
        curves.append({"budget": tag, "robust_em": metrics[f"robust_em_{tag}"]})

    save_json(cfg["logging"]["final_metrics_path"], metrics)
    save_json(os.path.join(cfg["logging"]["save_dir"], "eval_robust.json"), metrics)
    if curves:
        csv_path = os.path.join(cfg["logging"]["save_dir"], "curves_budget.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("budget,robust_em\n")
            for row in curves:
                f.write(f"{row['budget']},{row['robust_em']:.6f}\n")
    metrics_logger.log({"event": "eval_complete", **metrics})
    metrics_logger.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()
    eval_generation(args.config, args.ckpt, overrides=args.override)
