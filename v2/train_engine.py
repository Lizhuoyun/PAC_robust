"""
Unified training engine — model-agnostic, trainer-agnostic.

Usage:
    trainer_step = make_trainer("r3f_plugin", gamma=0.3, cfg=cfg)
    ckpt_dir = run_training(trainer_step, cfg, seed=42, tag="r3f_plugin_q25_s42")
"""
import os, math, random, json
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as _cfg_mod
from data    import load_data
from models  import load_for_training, tokenise_batch, extract_class_logits, get_label_ids


def run_training(trainer_step, cfg: dict, seed: int, tag: str) -> str:
    """
    Train model with the given trainer_step for one (method, seed).
    tag   : unique identifier, used as checkpoint folder name.
    Returns path to saved LoRA adapter.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    rng    = random.Random(seed)
    device = cfg["device"]

    print(f"\n{'='*64}")
    print(f"  Training  tag={tag}  seed={seed}")
    print(f"{'='*64}")

    data       = load_data(cfg, seed=seed)
    model, tok = load_for_training(cfg)
    label_ids  = get_label_ids(tok, cfg["label_chars"])

    # ── Optimizer & scheduler ──────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(params, lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])

    bs          = cfg["batch_size"]
    steps_epoch = math.ceil(len(data["train"]) / bs)
    total_steps = steps_epoch * cfg["num_epochs"]
    warmup      = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        return max(0.0, 1.0 - (s - warmup) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    save_dir     = os.path.join(_cfg_mod.CKPT_DIR, tag)
    best_val     = float("inf")
    grad_accum   = cfg["grad_accum"]
    global_step  = 0

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        model.train()
        items = list(data["train"])
        rng.shuffle(items)
        epoch_loss, n_b = 0.0, 0
        opt.zero_grad()

        pbar = tqdm(range(0, len(items), bs),
                    desc=f"Epoch {epoch+1}/{cfg['num_epochs']}", ncols=90)
        for step_i, start in enumerate(pbar):
            batch = items[start: start + bs]
            if not batch:
                continue

            loss, metrics, _, _, _ = trainer_step(
                model, tok, batch, cfg, rng, label_ids, device)

            (loss / grad_accum).backward()
            epoch_loss += loss.item()
            n_b        += 1

            if (step_i + 1) % grad_accum == 0 or (step_i + 1) == steps_epoch:
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{sched.get_last_lr()[0]:.2e}")

        avg = epoch_loss / max(1, n_b)
        print(f"  Epoch {epoch+1}  avg_loss={avg:.4f}")

        # ── Val eval (clean, quick) ────────────────────────────────────
        model.eval()
        val_loss = _quick_val(model, tok, data["val"][:200],
                              label_ids, cfg, device)
        print(f"  Val  loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)

    print(f"  Saved best ({best_val:.4f}) → {save_dir}")
    del model, opt, sched
    torch.cuda.empty_cache()
    return save_dir


@torch.no_grad()
def _quick_val(model, tok, items, label_ids, cfg, device) -> float:
    bs    = cfg["batch_size"]
    total = 0.0
    n     = 0
    for start in range(0, len(items), bs):
        batch   = items[start: start + bs]
        prompts = [e["prompt"]     for e in batch]
        lcs     = [e["label_char"] for e in batch]
        labs    = torch.tensor([e["label"] for e in batch], device=device)

        enc = tokenise_batch(tok, prompts, lcs, cfg["max_seq_len"])
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"])
        cl  = extract_class_logits(out.logits, enc["label_positions"], label_ids)
        total += F.cross_entropy(cl, labs).item() * len(batch)
        n     += len(batch)
    return total / max(1, n)
