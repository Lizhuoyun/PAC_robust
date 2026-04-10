import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.datasets import get_dataset
from src.data.prompts import PromptRenderer, PromptTemplate, extract_mc_fields
from src.data.perturbations import build_perturbed_fields_cache, load_perturbed_fields_cache, materialize_perturbed_fields
from src.losses.base_losses import classification_ce
from src.losses.r3f import r3f_kl_logits
from src.losses.smart import smart_kl
from src.losses.spectral import SpectralEMA, batch_transition_matrix, stability_kl, sample_lora_noise, apply_lora_noise, remove_lora_noise
from src.models.lora import apply_lora, iter_lora_params
from src.models.verbalizer import Verbalizer, margin_from_logits, resolve_verbalizer_ids
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
    clean_positions: torch.Tensor
    pert_input_ids: torch.Tensor
    pert_attention_mask: torch.Tensor
    pert_positions: torch.Tensor
    labels: torch.Tensor


def _encode_prompt_with_label(
    tokenizer,
    prompt: str,
    label_token: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode prompt and a single-token label in a truncation-safe way:
    always keep the label token as the last token in the sequence.
    Returns (input_ids, attention_mask, position) where position points to the
    last prompt token (the token that predicts the label token).
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_token, add_special_tokens=False)
    if len(label_ids) != 1:
        raise ValueError(f"label_token must be single-token after resolve_verbalizer_ids, got {label_token!r} -> {label_ids}")
    # Ensure we have room for the label token.
    keep = max(int(max_length) - 1, 1)
    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:]
    input_ids = torch.tensor(prompt_ids + label_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    pos = max(len(prompt_ids) - 1, 0)
    return input_ids, attention_mask, torch.tensor(pos, dtype=torch.long)


class PairedClassificationDataset(Dataset):
    def __init__(
        self,
        clean_data: List[Dict],
        pert_data: List[Dict],
        tokenizer,
        renderer: PromptRenderer,
        label_map: Dict[str, int],
        max_length: int,
        verbalizer_tokens: Dict[str, str],
    ):
        if len(clean_data) != len(pert_data):
            raise ValueError(f"clean/pert length mismatch: {len(clean_data)} vs {len(pert_data)}")
        self.clean_data = clean_data
        self.pert_data = pert_data
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.label_map = label_map
        self.max_length = max_length
        self.verbalizer_tokens = verbalizer_tokens

    def __len__(self) -> int:
        return len(self.clean_data)

    def __getitem__(self, idx: int) -> Batch:
        ex_c = self.clean_data[idx]
        ex_p = self.pert_data[idx]
        label = self.label_map[ex_c["label"]]
        verbalizer_token = self.verbalizer_tokens[ex_c["label"]]
        prompt_c = self.renderer.render(ex_c)
        prompt_p = self.renderer.render(ex_p)
        clean_ids, clean_mask, clean_pos = _encode_prompt_with_label(self.tokenizer, prompt_c, verbalizer_token, self.max_length)
        pert_ids, pert_mask, pert_pos = _encode_prompt_with_label(self.tokenizer, prompt_p, verbalizer_token, self.max_length)
        return Batch(
            clean_input_ids=clean_ids,
            clean_attention_mask=clean_mask,
            clean_positions=clean_pos,
            pert_input_ids=pert_ids,
            pert_attention_mask=pert_mask,
            pert_positions=pert_pos,
            labels=torch.tensor(label, dtype=torch.long),
        )


def collate_fn(batch: List[Batch], pad_token_id: int) -> Dict[str, torch.Tensor]:
    clean_input_ids = torch.nn.utils.rnn.pad_sequence([b.clean_input_ids for b in batch], batch_first=True, padding_value=pad_token_id)
    clean_attention_mask = torch.nn.utils.rnn.pad_sequence([b.clean_attention_mask for b in batch], batch_first=True, padding_value=0)
    clean_positions = torch.stack([b.clean_positions for b in batch])
    pert_input_ids = torch.nn.utils.rnn.pad_sequence([b.pert_input_ids for b in batch], batch_first=True, padding_value=pad_token_id)
    pert_attention_mask = torch.nn.utils.rnn.pad_sequence([b.pert_attention_mask for b in batch], batch_first=True, padding_value=0)
    pert_positions = torch.stack([b.pert_positions for b in batch])
    labels = torch.stack([b.labels for b in batch])
    return {
        "clean_input_ids": clean_input_ids,
        "clean_attention_mask": clean_attention_mask,
        "clean_positions": clean_positions,
        "pert_input_ids": pert_input_ids,
        "pert_attention_mask": pert_attention_mask,
        "pert_positions": pert_positions,
        "labels": labels,
    }


def train_classification(config_path: str, overrides: List[str] = None) -> None:
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_presets(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    template = PromptTemplate(answer_marker=cfg["dataset"].get("answer_marker", "Answer: "))
    data = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["train_split"])
    max_train = cfg.get("dataset", {}).get("max_train_examples") or cfg.get("dataset", {}).get("max_examples")
    if max_train:
        data = data[: int(max_train)]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_map = cfg["verbalizer"]
    label_map = {label: idx for idx, label in enumerate(verbalizer_map.keys())}
    labels = list(verbalizer_map.keys())
    resolved_map, verbalizer_ids, tmpl = resolve_verbalizer_ids(
        tokenizer,
        labels=labels,
        marker=template.answer_marker,
        templates=cfg["verbalizer"].get("templates", None) if isinstance(cfg["verbalizer"], dict) else None,
    )
    verbalizer = Verbalizer(resolved_map)

    renderer = PromptRenderer(template, labels)
    clean_fields = [extract_mc_fields(ex, labels) for ex in data]
    pert_cfg = dict(cfg["perturbation"])
    if cfg.get("augment", {}).get("enabled", False):
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
        # Backward-compat: old cache without source_idx should be 1:1 with clean_fields.
        clean_aligned = clean_fields
        if len(clean_aligned) != len(pert_fields):
            raise ValueError("cannot align clean/pert fields: missing source_idx in cache records and lengths differ")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    paired_dataset = PairedClassificationDataset(clean_aligned, pert_fields, tokenizer, renderer, label_map, cfg["max_length"], verbalizer.label_to_token)
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
        num_classes=len(labels),
        beta_ema=cfg["spectral"]["beta_ema"],
        device=device,
        n_refresh=cfg["spectral"]["n_refresh"],
    )

    # Spectral warmup: keep training stable by delaying spectral regularization and ramping it on.
    # We scale both alpha (sigma_max) and beta (stability_kl) by the same warmup schedule.
    # Config knobs (all optional):
    #   spectral.warmup_steps: int
    #   spectral.warmup_ratio: float in [0,1]
    #   spectral.ramp_steps: int
    #   spectral.ramp_ratio: float in [0,1]
    spec_cfg = cfg.get("spectral", {})
    warmup_steps = spec_cfg.get("warmup_steps", None)
    if warmup_steps is None:
        warmup_ratio = float(spec_cfg.get("warmup_ratio", 0.1) or 0.0)
        warmup_steps = int(round(warmup_ratio * total_steps))
    warmup_steps = max(int(warmup_steps), 0)
    ramp_steps = spec_cfg.get("ramp_steps", None)
    if ramp_steps is None:
        ramp_ratio = float(spec_cfg.get("ramp_ratio", 0.1) or 0.0)
        ramp_steps = int(round(ramp_ratio * total_steps))
    ramp_steps = max(int(ramp_steps), 1)

    def _spectral_scale(step_idx: int) -> float:
        if step_idx <= warmup_steps:
            return 0.0
        return min(1.0, float(step_idx - warmup_steps) / float(ramp_steps))

    def forward_logits(input_ids, attention_mask, positions, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        pos_logits = logits[batch, positions]
        return pos_logits.index_select(dim=-1, index=verbalizer_ids.to(device))

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
            clean_pos = batch["clean_positions"].to(device)
            pert_ids = batch["pert_input_ids"].to(device)
            pert_mask = batch["pert_attention_mask"].to(device)
            pert_pos = batch["pert_positions"].to(device)
            labels_t = batch["labels"].to(device)
            clean_logits = forward_logits(clean_ids, clean_mask, clean_pos)
            task_loss = classification_ce(clean_logits, labels_t)

            loss = task_loss
            extra = {}

            if (augment_enabled and cfg["augment"]["clean_ratio"] < 1) or cfg["spectral"]["enabled"]:
                pert_logits = forward_logits(pert_ids, pert_mask, pert_pos)
                if augment_enabled and cfg["augment"]["clean_ratio"] < 1:
                    pert_loss = classification_ce(pert_logits, labels_t)
                    loss = cfg["augment"]["clean_ratio"] * task_loss + (1.0 - cfg["augment"]["clean_ratio"]) * pert_loss
            if cfg.get("r3f", {}).get("enabled", False):
                embeds = model.get_input_embeddings()(clean_ids)
                noise = torch.randn_like(embeds) * cfg["r3f"]["noise_std"]
                noisy_logits = forward_logits(clean_ids, clean_mask, clean_pos, inputs_embeds=embeds + noise)
                use_spectral_guided = cfg["r3f"].get("spectral_guided", False)
                r3f = r3f_kl_logits(
                    clean_logits,
                    noisy_logits,
                    detach_target=cfg["r3f"].get("detach_target", True),
                    labels=labels_t if use_spectral_guided else None,
                    spectral_cfg=cfg["spectral"] if use_spectral_guided else None,
                )
                loss = loss + cfg["r3f"]["lambda"] * r3f
                extra["r3f"] = r3f.detach()
                # S-R3F: spectral loss on the same noisy point (spectral inside R3F)
                if cfg["r3f"].get("spectral_guided", False) and cfg.get("spectral"):
                    spec = cfg["spectral"]
                    alpha = spec.get("alpha", 0.1)
                    gamma = spec.get("gamma", 0.2)
                    tau = spec.get("tau", 0.1)
                    probs = F.softmax(noisy_logits, dim=-1)
                    margins = margin_from_logits(noisy_logits, labels_t)
                    batch_mat = batch_transition_matrix(probs, labels_t, margins, gamma, tau)
                    spectral_risk = batch_mat.sum(dim=0).max()
                    loss = loss + alpha * spectral_risk
                    extra["r3f_spectral"] = spectral_risk.detach()

            if cfg.get("smart", {}).get("enabled", False):
                smart = smart_kl(
                    model,
                    clean_ids,
                    clean_mask,
                    logits_fn=lambda ids, mask, inputs_embeds=None: forward_logits(ids, mask, clean_pos, inputs_embeds=inputs_embeds),
                    steps=cfg["smart"]["K"],
                    step_size=cfg["smart"]["step_size"],
                    epsilon=cfg["smart"]["eps"],
                    norm=cfg["smart"]["norm"],
                    detach_target=cfg["smart"].get("detach_target", True),
                    labels=labels_t,
                    spectral_cfg=cfg["spectral"] if cfg["smart"].get("spectral_guided", False) else None,
                )
                loss = loss + cfg["smart"]["lambda"] * smart
                extra["smart"] = smart.detach()

            if cfg["spectral"]["enabled"]:
                probs = F.softmax(pert_logits, dim=-1)
                margins = margin_from_logits(pert_logits, labels_t)
                batch_mat = batch_transition_matrix(probs, labels_t, margins, cfg["spectral"]["gamma"], cfg["spectral"]["tau"])
                spectral.update(batch_mat)
                sigma = spectral.sigma_max(cfg["spectral"]["t_pi"])
                scale = _spectral_scale(step)
                alpha_eff = cfg["spectral"]["alpha"] * scale
                loss = loss + alpha_eff * sigma
                extra["sigma_max"] = sigma.detach()
                extra["spectral_scale"] = torch.tensor(scale, device=device)
                extra["alpha_eff"] = torch.tensor(alpha_eff, device=device)

                if cfg["spectral"]["stability"]:
                    lora_params = list(iter_lora_params(model))
                    if lora_params:
                        noise = sample_lora_noise(lora_params, cfg["spectral"]["sigma_noise"])
                        apply_lora_noise(lora_params, noise)
                        noisy_logits = forward_logits(pert_ids, pert_mask, pert_pos)
                        remove_lora_noise(lora_params, noise)
                        stab = stability_kl(pert_logits, noisy_logits)
                        beta_eff = cfg["spectral"]["beta"] * scale
                        loss = loss + beta_eff * stab
                        extra["stab"] = stab.detach()
                        extra["beta_eff"] = torch.tensor(beta_eff, device=device)

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
            "verbalizer_template": tmpl,
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
    train_classification(args.config)
