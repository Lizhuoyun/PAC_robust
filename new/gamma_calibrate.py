"""
Gamma calibration: compute margin distribution on a perturbed validation set
from a trained Base-aug model, then return quantile-based gamma values.
"""
import os, sys, random, json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import get_cfg
from model_utils import (load_model_from_checkpoint, get_label_token_ids,
                         tokenize_for_classification, extract_class_logits)
from data_utils import load_task_data, _agnews_prompt, _arc_prompt, _sqa_prompt, _robot_prompt
from perturbations import random_perturbation
from plugin import compute_margins
from train import _process_multimodal_single


def _rebuild_perturbed(ex, ptxt, task, cfg):
    if task == "agnews":
        return _agnews_prompt(ptxt)
    elif task == "arc":
        return _arc_prompt(ptxt, ex.get("choices", ["","","",""]))
    elif task == "scienceqa":
        return _sqa_prompt(ptxt, ex.get("choices", []))
    elif task == "robot":
        return _robot_prompt(ptxt, cfg["label_names"])
    return ex["prompt"]


@torch.no_grad()
def calibrate_gamma(cfg: dict, checkpoint_dir: str, seed: int = 42,
                    quantiles=(0.10, 0.25, 0.50)):
    """
    Load a Base-aug model, forward perturbed val data, compute margin distribution,
    return {quantile: gamma_value}.
    """
    model, tokenizer, processor = load_model_from_checkpoint(cfg, checkpoint_dir)
    model.eval()
    label_token_ids = get_label_token_ids(tokenizer, cfg["label_chars"])
    device = next(model.parameters()).device
    is_mm = cfg["modality"] == "multimodal"
    rng = random.Random(seed)

    data = load_task_data(cfg, seed=seed)
    val_data = data["val"]

    all_margins = []
    bs = cfg["batch_size"] if not is_mm else 1

    for start in tqdm(range(0, len(val_data), bs), desc="Gamma calibration"):
        batch = val_data[start:start + bs]

        proc_batch = []
        for ex in batch:
            ptxt, pimg = random_perturbation(
                ex["text"], ex.get("image"), ex["task"], cfg["modality"],
                fill_fields=ex.get("fill_fields"), cfg=cfg, rng=rng)
            ex_copy = dict(ex)
            ex_copy["text"] = ptxt
            ex_copy["prompt"] = _rebuild_perturbed(ex, ptxt, ex["task"], cfg)
            if pimg is not None:
                ex_copy["image"] = pimg
            proc_batch.append(ex_copy)

        if is_mm:
            for ex in proc_batch:
                inp, lp = _process_multimodal_single(
                    processor, ex["prompt"], ex["label_char"],
                    ex.get("image"), device, cfg["max_seq_len"])
                out = model(**inp)
                cl = extract_class_logits(out.logits,
                                          torch.tensor([lp], device=device),
                                          label_token_ids)
                lab = torch.tensor([ex["label"]], device=device)
                m = compute_margins(cl, lab)
                all_margins.append(m.cpu())
        else:
            prompts = [e["prompt"] for e in proc_batch]
            lcs = [e["label_char"] for e in proc_batch]
            labs = torch.tensor([e["label"] for e in proc_batch], device=device)
            enc = tokenize_for_classification(tokenizer, prompts, lcs, cfg["max_seq_len"])
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cl = extract_class_logits(out.logits, enc["label_positions"], label_token_ids)
            m = compute_margins(cl, labs)
            all_margins.append(m.cpu())

    margins = torch.cat(all_margins).numpy()
    result = {}
    for q in quantiles:
        result[q] = float(np.quantile(margins, q))
    result["mean"] = float(margins.mean())
    result["std"] = float(margins.std())
    result["min"] = float(margins.min())
    result["max"] = float(margins.max())

    print(f"Margin distribution: mean={result['mean']:.4f} std={result['std']:.4f}")
    for q in quantiles:
        print(f"  q{int(q*100):02d} = {result[q]:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = get_cfg(args.task, device=args.device)
    ckpt = os.path.join(cfg["checkpoint_dir"], f"{args.task}_base_aug_s{args.seed}")
    result = calibrate_gamma(cfg, ckpt, seed=args.seed)
    print(json.dumps(result, indent=2))
