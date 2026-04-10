"""
Evaluation module.

evaluate_split(model, tok, data, cfg, ptype, gamma, seed)
    → dict of all metrics for one (model, perturbation) combination

evaluate_all(model, tok, data, cfg, gamma, seed)
    → list of per-perturbation metric dicts
"""
import math, random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models  import tokenise_batch, extract_class_logits, get_label_ids
from perturb import apply_perturbation
from plugin  import (compute_vwr_gamma, compute_sigma_max,
                     compute_fragile_ratio, compute_mean_gate)


@torch.no_grad()
def evaluate_split(model, tok, data: list, cfg: dict,
                   ptype: str = "clean",
                   gamma: float = 0.0,
                   seed: int = 42) -> dict:
    """
    Evaluate model on data with a given perturbation.

    Returns a dict with keys:
      accuracy, per_class_acc, worst_class_acc, worst_class_err,
      clean_to_robust_drop (filled in by caller),
      vwr_gamma, sigma_max, fragile_ratio, mean_gate,
      confusion, n_samples, perturbation
    """
    device     = next(model.parameters()).device
    label_ids  = get_label_ids(tok, cfg["label_chars"])
    K          = cfg["num_classes"]
    bs         = cfg["batch_size"]
    rng        = random.Random(seed)

    all_preds, all_labels, all_logits = [], [], []

    for start in tqdm(range(0, len(data), bs),
                      desc=f"Eval [{ptype}]", leave=False):
        batch = data[start: start + bs]

        if ptype != "clean":
            batch = [apply_perturbation(ex, ptype, cfg, rng) for ex in batch]

        prompts = [e["prompt"]     for e in batch]
        lcs     = [e["label_char"] for e in batch]
        labs    = torch.tensor([e["label"] for e in batch])

        enc = tokenise_batch(tok, prompts, lcs, cfg["max_seq_len"])
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"])
        cl  = extract_class_logits(out.logits,
                                   enc["label_positions"],
                                   label_ids).cpu()

        all_logits.append(cl)
        all_preds.append(cl.argmax(-1))
        all_labels.append(labs)

    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)
    logits  = torch.cat(all_logits, dim=0)

    # ── Overall accuracy ───────────────────────────────────────────────
    correct = (preds == labels).float()
    acc     = correct.mean().item()

    # ── Per-class accuracy ─────────────────────────────────────────────
    per_class = {}
    for c in range(K):
        m = (labels == c)
        per_class[c] = correct[m].mean().item() if m.any() else float("nan")

    valid = [v for v in per_class.values() if not math.isnan(v)]
    worst_acc = min(valid) if valid else float("nan")
    worst_err = 1.0 - worst_acc if not math.isnan(worst_acc) else float("nan")

    # ── Confusion matrix ───────────────────────────────────────────────
    confusion = torch.zeros(K, K, dtype=torch.long)
    for p, l in zip(preds, labels):
        confusion[p.item(), l.item()] += 1

    # ── Plugin-related metrics ─────────────────────────────────────────
    kappa = cfg["kappa"]
    vwr   = compute_vwr_gamma(logits, labels, gamma, kappa, K)
    smax  = compute_sigma_max(logits, labels, gamma, kappa, K)
    frag  = compute_fragile_ratio(logits, labels, gamma, kappa)
    mgate = compute_mean_gate(logits, labels, gamma, kappa)

    return dict(
        perturbation     = ptype,
        accuracy         = acc,
        per_class_acc    = per_class,
        worst_class_acc  = worst_acc,
        worst_class_err  = worst_err,
        vwr_gamma        = vwr,
        sigma_max        = smax,
        fragile_ratio    = frag,
        mean_gate        = mgate,
        confusion        = confusion.tolist(),
        n_samples        = len(labels),
    )


def evaluate_all(model, tok, data: list, cfg: dict,
                 gamma: float = 0.0, seed: int = 42) -> list:
    """
    Evaluate on clean + all perturbation types.
    Fills in 'clean_to_robust_drop' for each non-clean entry.
    Returns list of metric dicts.
    """
    ptypes  = ["clean"] + cfg["perturb_types"]
    results = []

    for pt in ptypes:
        m = evaluate_split(model, tok, data, cfg,
                           ptype=pt, gamma=gamma, seed=seed)
        results.append(m)
        print(f"    [{pt:>14}]  acc={m['accuracy']:.4f}  "
              f"worst={m['worst_class_acc']:.4f}  "
              f"VWR={m['vwr_gamma']:.3f}  σ_max={m['sigma_max']:.3f}")

    # Clean-to-robust drop per perturbation
    clean_acc = results[0]["accuracy"]
    for r in results[1:]:
        r["clean_to_robust_drop"] = clean_acc - r["accuracy"]

    return results
