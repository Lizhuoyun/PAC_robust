import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.datasets import get_dataset
from src.data.prompts import PromptRenderer, PromptTemplate, extract_gen_fields
from src.data.perturbations import build_perturbed_fields_cache, load_perturbed_fields_cache, materialize_perturbed_fields
from src.losses.base_losses import lm_nll
from src.losses.r3f import r3f_kl_logits
from src.losses.smart import smart_kl
from src.losses.spectral import SpectralEMA, batch_transition_matrix, stability_kl, sample_lora_noise, apply_lora_noise, remove_lora_noise
from src.models.lora import apply_lora, iter_lora_params
from src.logging.logger import ExperimentLogger
from src.train.utils import (
    apply_overrides,
    elapsed,
    get_device,
    load_config,
    resolve_torch_dtype,
    resolve_presets,
    save_config,
    save_json,
    set_seed,
    timer,
)


@dataclass
class Batch:
    clean_input_ids: torch.Tensor
    clean_attention_mask: torch.Tensor
    clean_labels: torch.Tensor
    pert_input_ids: torch.Tensor
    pert_attention_mask: torch.Tensor
    pert_labels: torch.Tensor


def _encode_prompt_answer(
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncation-safe encoding for generation training.
    We construct ids as prompt_ids + answer_ids (answer prefixed by a space),
    and truncate from the left of prompt to always keep the answer tokens.
    Returns (input_ids, attention_mask, labels) where labels mask prompt tokens with -100.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Match previous behavior: full = prompt + " " + answer
    answer_text = (" " + answer) if answer else ""
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) if answer_text else []

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    # Ensure answer is kept; if answer itself is longer than max_length, keep its tail.
    if len(answer_ids) >= max_length:
        prompt_ids = []
        answer_ids = answer_ids[-max_length:]
    else:
        keep_prompt = max_length - len(answer_ids)
        if len(prompt_ids) > keep_prompt:
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100
    return input_ids, attention_mask, labels


class PairedGenerationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, renderer: PromptRenderer, max_length: int):
        clean_data, pert_data = data
        if len(clean_data) != len(pert_data):
            raise ValueError(f"clean/pert length mismatch: {len(clean_data)} vs {len(pert_data)}")
        self.clean_data = clean_data
        self.pert_data = pert_data
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.clean_data)

    def __getitem__(self, idx: int) -> Batch:
        ex_c = self.clean_data[idx]
        ex_p = self.pert_data[idx]
        answer = ex_c["answer"]
        # Ensure paired samples share the same gold answer.
        if ex_p.get("answer") is not None:
            answer = ex_p.get("answer", answer)
        prompt_c = self.renderer.render(ex_c)
        prompt_p = self.renderer.render(ex_p)
        c_ids, c_mask, c_labels = _encode_prompt_answer(self.tokenizer, prompt_c, answer, self.max_length)
        p_ids, p_mask, p_labels = _encode_prompt_answer(self.tokenizer, prompt_p, answer, self.max_length)
        return Batch(
            clean_input_ids=c_ids,
            clean_attention_mask=c_mask,
            clean_labels=c_labels,
            pert_input_ids=p_ids,
            pert_attention_mask=p_mask,
            pert_labels=p_labels,
        )


def collate_fn(batch: List[Batch], pad_token_id: int) -> Dict[str, torch.Tensor]:
    clean_input_ids = torch.nn.utils.rnn.pad_sequence([b.clean_input_ids for b in batch], batch_first=True, padding_value=pad_token_id)
    clean_attention_mask = torch.nn.utils.rnn.pad_sequence([b.clean_attention_mask for b in batch], batch_first=True, padding_value=0)
    clean_labels = torch.nn.utils.rnn.pad_sequence([b.clean_labels for b in batch], batch_first=True, padding_value=-100)
    pert_input_ids = torch.nn.utils.rnn.pad_sequence([b.pert_input_ids for b in batch], batch_first=True, padding_value=pad_token_id)
    pert_attention_mask = torch.nn.utils.rnn.pad_sequence([b.pert_attention_mask for b in batch], batch_first=True, padding_value=0)
    pert_labels = torch.nn.utils.rnn.pad_sequence([b.pert_labels for b in batch], batch_first=True, padding_value=-100)
    return {
        "clean_input_ids": clean_input_ids,
        "clean_attention_mask": clean_attention_mask,
        "clean_labels": clean_labels,
        "pert_input_ids": pert_input_ids,
        "pert_attention_mask": pert_attention_mask,
        "pert_labels": pert_labels,
    }


def _candidate_probs(logits: torch.Tensor, gold_ids: torch.Tensor, top_k: int) -> torch.Tensor:
    batch, vocab = logits.size()
    masked = logits.clone()
    masked[torch.arange(batch, device=logits.device), gold_ids] = -1e9
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


def train_generation(config_path: str, overrides: List[str] = None) -> None:
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_presets(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    template = PromptTemplate(final_marker=cfg["dataset"].get("final_marker", "Final answer: "))
    data = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["train_split"])
    max_train = cfg.get("dataset", {}).get("max_train_examples") or cfg.get("dataset", {}).get("max_examples")
    if max_train:
        data = data[: int(max_train)]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    renderer = PromptRenderer(template)
    clean_fields = [extract_gen_fields(ex) for ex in data]
    pert_cfg = cfg["perturbation"]
    if cfg.get("augment", {}).get("enabled", False):
        pert_cfg = dict(pert_cfg)
        pert_cfg["budget"] = cfg["augment"].get("train_budget")
    cache_path = build_perturbed_fields_cache(
        clean_fields,
        dataset_name=cfg["dataset"]["name"],
        split=cfg["dataset"]["train_split"],
        config=pert_cfg,
        seed=cfg["seed"],
        n_variants=cfg.get("augment", {}).get("n_perturb_per_clean", 1),
        cache_root=cfg["perturbation"].get("cache_root", "cache/perturb_fields"),
    )
    cache_records = load_perturbed_fields_cache(cache_path)
    pert_fields = materialize_perturbed_fields(clean_fields, cache_records)
    # Expand clean fields to align with cache_records (n_variants may repeat each source_idx).
    if cache_records and "source_idx" in cache_records[0]:
        clean_aligned = [clean_fields[int(rec["source_idx"])] for rec in cache_records]
    else:
        clean_aligned = clean_fields
        if len(clean_aligned) != len(pert_fields):
            raise ValueError("cannot align clean/pert fields: missing source_idx in cache records and lengths differ")

    paired_dataset = PairedGenerationDataset([clean_aligned, pert_fields], tokenizer, renderer, cfg["max_length"])
    loader = DataLoader(
        paired_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
    )
    augment_enabled = cfg.get("augment", {}).get("enabled", False)

    torch_dtype = resolve_torch_dtype(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if cfg["finetune_mode"] == "lora":
        model = apply_lora(
            model,
            target_modules=cfg["lora"]["target_modules"],
            r=cfg["lora"]["r"],
            alpha=cfg["lora"]["alpha"],
            dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
        )
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    total_steps = len(loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg["optimizer"]["warmup_steps"], total_steps)

    metrics_logger = ExperimentLogger(cfg, cfg["logging"]["metrics_path"])
    save_config(os.path.join(cfg["logging"]["save_dir"], "config_resolved.yaml"), cfg)
    start = timer()
    epoch_times = []

    spectral = SpectralEMA(
        num_classes=cfg["spectral"]["top_k"] + 1,
        beta_ema=cfg["spectral"]["beta_ema"],
        device=device,
        n_refresh=cfg["spectral"]["n_refresh"],
    )

    step = 0
    base_obj = cfg["objective"]["name"]
    forward_passes = 1
    if augment_enabled and cfg["augment"]["clean_ratio"] < 1:
        forward_passes += 1
    if cfg.get("r3f", {}).get("enabled", False):
        forward_passes += 1
    if cfg.get("smart", {}).get("enabled", False):
        forward_passes += cfg["smart"]["K"] + 1
    if cfg["spectral"]["enabled"]:
        # spectral needs perturbed logits
        forward_passes += 1
    if cfg["spectral"]["stability"]:
        forward_passes += 1
    for epoch in range(cfg["epochs"]):
        epoch_start = timer()
        for batch in loader:
            step += 1
            clean_ids = batch["clean_input_ids"].to(device)
            clean_mask = batch["clean_attention_mask"].to(device)
            clean_labels = batch["clean_labels"].to(device)
            pert_ids = batch["pert_input_ids"].to(device)
            pert_mask = batch["pert_attention_mask"].to(device)
            pert_labels = batch["pert_labels"].to(device)

            outputs = model(input_ids=clean_ids, attention_mask=clean_mask)
            logits = outputs.logits
            task_loss = lm_nll(logits, clean_labels)
            loss = task_loss
            extra = {}

            pert_logits_full = None
            if (augment_enabled and cfg["augment"]["clean_ratio"] < 1) or cfg["spectral"]["enabled"]:
                pert_logits_full = model(input_ids=pert_ids, attention_mask=pert_mask).logits
                if augment_enabled and cfg["augment"]["clean_ratio"] < 1:
                    pert_loss = lm_nll(pert_logits_full, pert_labels)
                    loss = cfg["augment"]["clean_ratio"] * task_loss + (1.0 - cfg["augment"]["clean_ratio"]) * pert_loss

            if cfg.get("r3f", {}).get("enabled", False):
                embeds = model.get_input_embeddings()(clean_ids)
                noise = torch.randn_like(embeds) * cfg["r3f"]["noise_std"]
                noisy_logits = model(inputs_embeds=embeds + noise, attention_mask=clean_mask).logits
                mask = (clean_labels != -100).float()
                r3f = r3f_kl_logits(logits, noisy_logits, mask=mask, detach_target=cfg["r3f"].get("detach_target", True))
                loss = loss + cfg["r3f"]["lambda"] * r3f
                extra["r3f"] = r3f.detach()

            if cfg.get("smart", {}).get("enabled", False):
                def logits_fn(ids, mask, inputs_embeds=None):
                    if inputs_embeds is not None:
                        return model(inputs_embeds=inputs_embeds, attention_mask=mask).logits
                    return model(input_ids=ids, attention_mask=mask).logits

                # In generation, labels are the full input_ids (shifted in loss)
                # But for spectral guidance, we need to know where the actual answer is
                smart = smart_kl(
                    model,
                    clean_ids,
                    clean_mask,
                    logits_fn=logits_fn,
                    steps=cfg["smart"]["K"],
                    step_size=cfg["smart"]["step_size"],
                    epsilon=cfg["smart"]["eps"],
                    norm=cfg["smart"]["norm"],
                    mask=(clean_labels != -100).float(),
                    detach_target=cfg["smart"].get("detach_target", True),
                    labels=clean_labels if cfg["smart"].get("spectral_guided", False) else None,
                    spectral_cfg=cfg["spectral"] if cfg["smart"].get("spectral_guided", False) else None,
                )
                loss = loss + cfg["smart"]["lambda"] * smart
                extra["smart"] = smart.detach()

            if cfg["spectral"]["enabled"]:
                if pert_logits_full is None:
                    pert_logits_full = model(input_ids=pert_ids, attention_mask=pert_mask).logits
                with torch.no_grad():
                    mask = pert_labels != -100
                    if cfg["generation"].get("positions", "answer") == "uniform":
                        mask = _sample_positions(mask, cfg["generation"].get("sample_n", 0))
                    pos_indices = mask.nonzero(as_tuple=False)
                if pos_indices.numel() > 0:
                    logits_pos = pert_logits_full[pos_indices[:, 0], pos_indices[:, 1]]
                    gold_ids = pert_labels[pos_indices[:, 0], pos_indices[:, 1]]
                    probs = _candidate_probs(logits_pos, gold_ids, cfg["spectral"]["top_k"])
                    gold_logit = logits_pos.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
                    max_comp = logits_pos.topk(cfg["spectral"]["top_k"], dim=-1).values.max(dim=1).values
                    margins = gold_logit - max_comp
                    labels_local = torch.zeros(probs.size(0), dtype=torch.long, device=device)
                    batch_mat = batch_transition_matrix(
                        probs,
                        labels_local,
                        margins,
                        cfg["spectral"]["gamma"],
                        cfg["spectral"]["tau"],
                    )
                    spectral.update(batch_mat)
                    sigma = spectral.sigma_max(cfg["spectral"]["t_pi"])
                    loss = loss + cfg["spectral"]["alpha"] * sigma
                    extra["sigma_max"] = sigma.detach()

                    if cfg["spectral"]["stability"]:
                        lora_params = list(iter_lora_params(model))
                        if lora_params:
                            noise = sample_lora_noise(lora_params, cfg["spectral"]["sigma_noise"])
                            apply_lora_noise(lora_params, noise)
                            noisy_logits = model(input_ids=pert_ids, attention_mask=pert_mask).logits
                            remove_lora_noise(lora_params, noise)
                            logits_pos_noisy = noisy_logits[pos_indices[:, 0], pos_indices[:, 1]]
                            stab = stability_kl(logits_pos, logits_pos_noisy)
                            loss = loss + cfg["spectral"]["beta"] * stab
                            extra["stab"] = stab.detach()

            if cfg["spectral"]["eta"] > 0:
                l2 = torch.tensor(0.0, device=device)
                for p in model.parameters():
                    if p.requires_grad:
                        l2 = l2 + (p ** 2).sum()
                loss = loss + cfg["spectral"]["eta"] * l2
                extra["l2"] = l2.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % cfg["logging"]["log_every"] == 0:
                lr = scheduler.get_last_lr()[0] if scheduler is not None else cfg["optimizer"]["lr"]
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "task_loss": task_loss.item(),
                    "lr": lr,
                    "elapsed": elapsed(start),
                    "forward_passes": forward_passes,
                }
                metrics.update({k: v.item() for k, v in extra.items()})
                metrics_logger.log(metrics, step=step)
        epoch_times.append(elapsed(epoch_start))

    save_dir = cfg["logging"]["save_dir"]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_json(
        cfg["logging"]["final_metrics_path"],
        {
            "status": "done",
            "steps": step,
            "epoch_times": epoch_times,
            "total_time": elapsed(start),
            "forward_passes": forward_passes,
            "augment": cfg.get("augment", {}),
            "r3f": cfg.get("r3f", {}),
            "smart": cfg.get("smart", {}),
            "spectral": cfg.get("spectral", {}),
        },
    )
    metrics_logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_generation(args.config)
