"""
Gamma calibration.

Given a Base-aug checkpoint, evaluate on the perturbed validation set,
collect true-label margins, and return quantile-based gamma values.
"""
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models  import load_from_checkpoint, tokenise_batch, extract_class_logits, get_label_ids
from data    import load_data
from perturb import apply_perturbation
from plugin  import compute_margins


@torch.no_grad()
def calibrate_gamma(cfg: dict, ckpt_dir: str,
                    seed: int = 42) -> dict:
    """
    Loads checkpoint, runs inference on val-set with all perturbation types,
    collects margins, returns {quantile: gamma_value}.

    Example: {0.10: 0.12, 0.25: 0.31, 0.50: 0.55}
    """
    model, tok = load_from_checkpoint(cfg, ckpt_dir)
    model.eval()
    device    = next(model.parameters()).device
    label_ids = get_label_ids(tok, cfg["label_chars"])
    K         = cfg["num_classes"]
    bs        = cfg["batch_size"]
    rng       = random.Random(seed)

    data = load_data(cfg, seed=seed)
    val  = data["val"]

    all_margins = []

    for ptype in cfg["perturb_types"]:
        for start in tqdm(range(0, len(val), bs),
                          desc=f"Calibrate [{ptype}]", leave=False):
            batch = [apply_perturbation(ex, ptype, cfg, rng)
                     for ex in val[start: start + bs]]
            prompts = [e["prompt"]     for e in batch]
            lcs     = [e["label_char"] for e in batch]
            labs    = torch.tensor([e["label"] for e in batch], device=device)

            enc = tokenise_batch(tok, prompts, lcs, cfg["max_seq_len"])
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
            cl  = extract_class_logits(out.logits, enc["label_positions"],
                                       label_ids)
            m   = compute_margins(cl, labs)
            all_margins.append(m.cpu())

    margins_cat = torch.cat(all_margins)
    quantiles   = cfg["gamma_quantiles"]
    gamma_vals  = {}
    for q in quantiles:
        gamma_vals[q] = float(torch.quantile(margins_cat,
                                              torch.tensor(q)).item())
        print(f"  gamma(q{int(q*100):02d}) = {gamma_vals[q]:.4f}")

    del model
    torch.cuda.empty_cache()
    return gamma_vals
