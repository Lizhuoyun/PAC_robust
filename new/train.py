"""
Unified training loop for Base-clean / Base-aug / Base-aug+Plugin.
Supports text-only and multimodal (VLM) tasks.
"""
import os, sys, json, random, time, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import get_cfg
from model_utils import (load_model_and_tokenizer, get_label_token_ids,
                         tokenize_for_classification, extract_class_logits)
from data_utils import load_task_data
from perturbations import apply_perturbation, random_perturbation
from plugin import plugin_loss


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal helpers (process one sample at a time via the VLM processor)
# ──────────────────────────────────────────────────────────────────────────────

def _process_multimodal_single(processor, prompt: str, label_char: str,
                               image, device, max_len=384):
    """Build input_ids / pixel_values for one multimodal example."""
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": prompt})

    prompt_text = processor.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    full_text = prompt_text + label_char

    inputs = processor(text=[full_text], images=[image] if image is not None else None,
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        prompt_inputs = processor(text=[prompt_text],
                                  images=[image] if image is not None else None,
                                  return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_len)
    label_pos = prompt_inputs["input_ids"].shape[1]  # label is the next token
    return inputs, label_pos


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def train_step_text(model, tokenizer, examples, mode, label_token_ids, cfg,
                    rng, gamma=None):
    """
    One gradient-accumulation macro-step for TEXT tasks.
    examples: list[dict]
    mode: 'base_clean' | 'base_aug' | 'plugin'
    Returns (loss_value, metrics_dict).
    """
    device = next(model.parameters()).device
    num_classes = cfg["num_classes"]
    prompts_clean = [ex["prompt"] for ex in examples]
    label_chars = [ex["label_char"] for ex in examples]
    labels = torch.tensor([ex["label"] for ex in examples], device=device)

    enc_clean = tokenize_for_classification(tokenizer, prompts_clean, label_chars,
                                            max_len=cfg["max_seq_len"])
    enc_clean = {k: v.to(device) for k, v in enc_clean.items()}

    out_clean = model(input_ids=enc_clean["input_ids"],
                      attention_mask=enc_clean["attention_mask"])
    cl_logits = extract_class_logits(out_clean.logits, enc_clean["label_positions"],
                                     label_token_ids)
    ce_clean = F.cross_entropy(cl_logits, labels)

    if mode == "base_clean":
        return ce_clean, {"ce": ce_clean.item()}

    # Perturbed forward
    perturbed_prompts = []
    for ex in examples:
        ptxt, _ = random_perturbation(
            ex["text"], None, ex["task"], "text",
            fill_fields=ex.get("fill_fields"), cfg=cfg, rng=rng,
        )
        # rebuild prompt from perturbed text
        from data_utils import _agnews_prompt, _arc_prompt
        if ex["task"] == "agnews":
            perturbed_prompts.append(_agnews_prompt(ptxt))
        elif ex["task"] == "arc":
            perturbed_prompts.append(_arc_prompt(ptxt, ex.get("choices", ["","","",""])))
        else:
            perturbed_prompts.append(ex["prompt"])  # fallback

    enc_pert = tokenize_for_classification(tokenizer, perturbed_prompts, label_chars,
                                           max_len=cfg["max_seq_len"])
    enc_pert = {k: v.to(device) for k, v in enc_pert.items()}
    out_pert = model(input_ids=enc_pert["input_ids"],
                     attention_mask=enc_pert["attention_mask"])
    pt_logits = extract_class_logits(out_pert.logits, enc_pert["label_positions"],
                                     label_token_ids)
    ce_pert = F.cross_entropy(pt_logits, labels)
    base_loss = (ce_clean + ce_pert) / 2

    if mode == "base_aug":
        return base_loss, {"ce": base_loss.item()}

    # Plugin mode
    assert gamma is not None, "gamma must be set for plugin mode"
    rs, rst, reg = plugin_loss(cl_logits.detach(), pt_logits, labels,
                               gamma=gamma, kappa=cfg["kappa"],
                               num_classes=num_classes,
                               alpha=cfg["alpha"], beta=cfg["beta"])
    total = base_loss + reg
    return total, {"ce": base_loss.item(), "r_spec": rs.item(), "r_stab": rst.item()}


def train_step_multimodal(model, processor, tokenizer, examples, mode,
                          label_token_ids, cfg, rng, gamma=None):
    """
    One macro-step for MULTIMODAL tasks — processes one sample at a time
    and accumulates gradients.
    """
    device = next(model.parameters()).device
    num_classes = cfg["num_classes"]
    n = len(examples)

    all_cl_logits, all_pt_logits, all_labels = [], [], []
    total_ce_clean = 0.0
    total_ce_pert = 0.0

    for ex in examples:
        label = torch.tensor([ex["label"]], device=device)
        lc = ex["label_char"]

        # Clean
        inp_c, lpos_c = _process_multimodal_single(
            processor, ex["prompt"], lc, ex["image"], device, cfg["max_seq_len"])
        out_c = model(**inp_c)
        lpos_t = torch.tensor([lpos_c], device=device)
        cl_log = extract_class_logits(out_c.logits, lpos_t,
                                      label_token_ids)
        ce_c = F.cross_entropy(cl_log, label)
        all_cl_logits.append(cl_log)
        total_ce_clean += ce_c / n

        if mode == "base_clean":
            (ce_c / n).backward()
            continue

        # Perturbed
        ptxt, pimg = random_perturbation(
            ex["text"], ex["image"], ex["task"], cfg["modality"],
            fill_fields=ex.get("fill_fields"), cfg=cfg, rng=rng,
        )
        from data_utils import _sqa_prompt, _robot_prompt
        if ex["task"] == "scienceqa":
            p_prompt = _sqa_prompt(ptxt, ex.get("choices", []))
        elif ex["task"] == "robot":
            p_prompt = _robot_prompt(ptxt, cfg["label_names"])
        else:
            p_prompt = ex["prompt"]

        inp_p, lpos_p = _process_multimodal_single(
            processor, p_prompt, lc, pimg, device, cfg["max_seq_len"])
        out_p = model(**inp_p)
        lpos_pt = torch.tensor([lpos_p], device=device)
        pt_log = extract_class_logits(out_p.logits, lpos_pt, label_token_ids)
        ce_p = F.cross_entropy(pt_log, label)
        all_pt_logits.append(pt_log)
        all_labels.append(label)
        total_ce_pert += ce_p / n

    if mode == "base_clean":
        return total_ce_clean, {"ce": total_ce_clean if isinstance(total_ce_clean, float) else total_ce_clean.item()}

    base_loss = (total_ce_clean + total_ce_pert) / 2

    if mode == "base_aug":
        base_loss.backward()
        return base_loss.item(), {"ce": base_loss.item()}

    # Plugin
    assert gamma is not None
    pt_cat = torch.cat(all_pt_logits, dim=0)
    cl_cat = torch.cat(all_cl_logits, dim=0).detach()
    lab_cat = torch.cat(all_labels, dim=0)
    rs, rst, reg = plugin_loss(cl_cat, pt_cat, lab_cat,
                               gamma=gamma, kappa=cfg["kappa"],
                               num_classes=num_classes,
                               alpha=cfg["alpha"], beta=cfg["beta"])
    total = base_loss + reg
    total.backward()
    return total.item(), {"ce": base_loss.item(), "r_spec": rs.item(), "r_stab": rst.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Full training run
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, mode: str, gamma: float = None, seed: int = 42):
    """
    Train a model for one (task, mode, seed) configuration.
    mode: 'base_clean' | 'base_aug' | 'plugin'
    Returns path to saved LoRA adapter.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)

    print(f"\n{'='*60}")
    print(f"Training: task={cfg['task_name']}  mode={mode}  seed={seed}")
    print(f"{'='*60}")

    data = load_task_data(cfg, seed=seed)
    print(f"  Data: train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")

    model, tokenizer, processor = load_model_and_tokenizer(cfg)
    label_token_ids = get_label_token_ids(tokenizer, cfg["label_chars"])
    device = next(model.parameters()).device

    is_mm = cfg["modality"] == "multimodal"
    bs = cfg["batch_size"] if not is_mm else 2  # smaller batch for multimodal

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    num_steps_per_epoch = math.ceil(len(data["train"]) / bs)
    total_steps = num_steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    save_dir = os.path.join(cfg["checkpoint_dir"],
                            f"{cfg['task_name']}_{mode}_s{seed}")

    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_items = list(data["train"])
        rng.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(range(0, len(train_items), bs),
                    desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
        for start in pbar:
            batch = train_items[start:start + bs]
            if not batch:
                continue

            optimizer.zero_grad()
            if is_mm:
                loss_val, metrics = train_step_multimodal(
                    model, processor, tokenizer, batch, mode,
                    label_token_ids, cfg, rng, gamma)
            else:
                loss, metrics = train_step_text(
                    model, tokenizer, batch, mode,
                    label_token_ids, cfg, rng, gamma)
                loss.backward()
                loss_val = loss.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Quick val loss (clean only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vstart in range(0, min(len(data["val"]), 200), bs):
                vbatch = data["val"][vstart:vstart + bs]
                if is_mm:
                    for ex in vbatch:
                        lab = torch.tensor([ex["label"]], device=device)
                        inp, lp = _process_multimodal_single(
                            processor, ex["prompt"], ex["label_char"],
                            ex["image"], device, cfg["max_seq_len"])
                        out = model(**inp)
                        cl = extract_class_logits(out.logits,
                                                  torch.tensor([lp], device=device),
                                                  label_token_ids)
                        val_loss += F.cross_entropy(cl, lab).item()
                else:
                    prompts = [e["prompt"] for e in vbatch]
                    lcs = [e["label_char"] for e in vbatch]
                    labs = torch.tensor([e["label"] for e in vbatch], device=device)
                    enc = tokenize_for_classification(tokenizer, prompts, lcs,
                                                      cfg["max_seq_len"])
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out = model(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"])
                    cl = extract_class_logits(out.logits, enc["label_positions"],
                                              label_token_ids)
                    val_loss += F.cross_entropy(cl, labs).item() * len(vbatch)

        val_loss /= max(1, min(len(data["val"]), 200))
        print(f"  Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f"  Best val loss: {best_val_loss:.4f}  Saved to {save_dir}")

    # Free memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return save_dir


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", required=True, choices=["base_clean", "base_aug", "plugin"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    overrides = dict(device=args.device)
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.beta is not None:
        overrides["beta"] = args.beta

    cfg = get_cfg(args.task, **overrides)
    train(cfg, mode=args.mode, gamma=args.gamma, seed=args.seed)
