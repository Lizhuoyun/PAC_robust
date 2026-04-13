"""
Unified evaluation for ScienceQA LoRA / Plugin-LoRA checkpoints.
Computes: Clean/SA/PA accuracy, worst-class, VWR_gamma, sigma_max.

Usage:
  python sqa_unified_eval.py --ckpt_dir <path> --gamma 0.15 --gpu 0
  python sqa_unified_eval.py --ckpt_dir <path> --gamma 0.15 --gpu 0 --eval_types clean,sa,pa
"""
import os, sys, json, re, copy, argparse, random, itertools
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "/LOCAL2/zhuoyun/PAC_robust/ARDS")
sys.path.insert(0, "/LOCAL2/zhuoyun/PAC_robust/new")

os.environ["HF_HOME"] = "/LOCAL2/zhuoyun/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/LOCAL2/zhuoyun/hf_cache"

from plugin import compute_vwr_gamma, compute_sigma_max, compute_margins, compute_gates

MODEL_PATH = "/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b"
TEST_DATA = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/llava_test_CQM-A.json"
IMAGE_FOLDER = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/images"

LABEL_TOKEN_IDS = [319, 350, 315, 360]
LABEL_CHARS = ["A", "B", "C", "D"]
NUM_CLASSES = 4
IGNORE_INDEX = -100

SA_MAP = {"A": "Q", "B": "W", "C": "E", "D": "R"}


def load_model_with_lora(ckpt_dir, device):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from peft import PeftModel

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name)
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer, image_processor


def load_test_data():
    data = json.load(open(TEST_DATA))
    for d in data:
        ans = d["conversations"][1]["value"].strip()
        if "The answer is " in ans:
            letter = ans.split("The answer is ")[-1].rstrip(".")
        else:
            letter = ans.strip().rstrip(".")
        d["label"] = LABEL_CHARS.index(letter) if letter in LABEL_CHARS else 0

        q = d["conversations"][0]["value"]
        opts = re.findall(r'\(([A-E])\)', q)
        d["num_options"] = len(opts)
    return data


def apply_sa_to_text(text):
    for orig, repl in SA_MAP.items():
        text = re.sub(rf'\(({orig})\)', f'({repl})', text)
        text = re.sub(rf'\b{orig}\.\s', f'{repl}. ', text)
    return text


def apply_pa_to_item(item, rng):
    """Position Attack: permute the order of answer options."""
    q = item["conversations"][0]["value"]
    lines = q.splitlines()
    option_lines = []
    for idx, line in enumerate(lines):
        m = re.match(r'^(\s*)(?:\(([A-E])\)|([A-E])\.)\s*(.*)$', line)
        if not m:
            continue
        letter = m.group(2) or m.group(3)
        option_lines.append({
            "idx": idx,
            "indent": m.group(1),
            "style": "paren" if m.group(2) else "dot",
            "letter": letter,
            "text": m.group(4),
        })

    if len(option_lines) < 2 or len(option_lines) > len(LABEL_CHARS):
        return item, item["label"]

    n = len(option_lines)
    perm = list(range(n))
    rng.shuffle(perm)

    orig_label = item["label"]
    new_label = perm.index(orig_label) if orig_label < n else orig_label

    for i in range(n):
        new_letter = LABEL_CHARS[i]
        target = option_lines[i]
        source = option_lines[perm[i]]
        if target["style"] == "paren":
            lines[target["idx"]] = f"{target['indent']}({new_letter}) {source['text']}"
        else:
            lines[target["idx"]] = f"{target['indent']}{new_letter}. {source['text']}"

    new_item = copy.deepcopy(item)
    new_item["conversations"][0]["value"] = "\n".join(lines)
    new_item["conversations"][1]["value"] = f"The answer is {LABEL_CHARS[new_label]}."
    return new_item, new_label


def prepare_single(item, tokenizer, image_processor, model, device,
                   attack_type="clean", rng=None):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from PIL import Image

    effective_label = item["label"]

    if attack_type == "sa":
        conv_item = copy.deepcopy(item)
        conv_item["conversations"][0]["value"] = apply_sa_to_text(
            conv_item["conversations"][0]["value"])
    elif attack_type == "pa":
        conv_item, effective_label = apply_pa_to_item(item, rng)
    else:
        conv_item = item

    conv = conv_templates["vicuna_v1"].copy()
    human_msg = conv_item["conversations"][0]["value"]
    gpt_msg = conv_item["conversations"][1]["value"]

    if "<image>" in human_msg:
        human_clean = human_msg.replace("<image>", "").strip()
        human_formatted = DEFAULT_IMAGE_TOKEN + "\n" + human_clean
    else:
        human_formatted = human_msg

    conv.append_message(conv.roles[0], human_formatted)
    conv.append_message(conv.roles[1], gpt_msg)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).to(device)

    images = None
    if "image" in item:
        img_path = os.path.join(IMAGE_FOLDER, item["image"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images = process_images([img], image_processor, model.config)
            images = images.half().to(device)

    labels = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    parts = prompt.split(sep)
    if len(parts) >= 2:
        inst_text = parts[0] + sep
        inst_ids = tokenizer_image_token(inst_text, tokenizer, IMAGE_TOKEN_INDEX,
                                         return_tensors='pt')
        labels[0, :len(inst_ids)] = IGNORE_INDEX

    return input_ids, labels, images, effective_label


def extract_class_logits(logits, input_ids, device):
    """Extract class logits at the position just before the answer token."""
    B, T, V = logits.shape
    label_ids_t = torch.tensor(LABEL_TOKEN_IDS, device=device)
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    search_len = min(len(ids), T)
    for t in range(search_len - 1, max(search_len - 30, -1), -1):
        tok = ids[t].item()
        if tok in LABEL_TOKEN_IDS:
            pos = t - 1
            if 0 <= pos < T:
                return logits[0, pos, label_ids_t].unsqueeze(0)
    pos = T - 2
    return logits[0, pos, label_ids_t].unsqueeze(0)


@torch.no_grad()
def evaluate(ckpt_dir, gamma, device_id=0, eval_types=None, kappa=0.5, seed=42):
    from sqa_plugin_trainer import (
        BATCH_SIZE,
        apply_position_attack,
        apply_symbol_attack,
        extract_answer_logits as extract_answer_logits_batched,
        get_label_token_ids,
        prepare_batch,
    )

    if eval_types is None:
        eval_types = ["clean", "sa", "pa"]

    device = f"cuda:{device_id}"
    model, tokenizer, image_processor = load_model_with_lora(ckpt_dir, device)
    test_data = load_test_data()
    rng = random.Random(seed)
    label_token_ids = get_label_token_ids(tokenizer)

    results = {}
    for etype in eval_types:
        all_logits = []
        all_preds = []
        all_labels = []

        for start in tqdm(range(0, len(test_data), BATCH_SIZE),
                          desc=f"Eval {etype}", leave=False):
            batch_items = copy.deepcopy(test_data[start:start + BATCH_SIZE])
            if etype == "sa":
                batch_items = apply_symbol_attack(batch_items)
            elif etype == "pa":
                batch_items = apply_position_attack(batch_items, rng)

            text_only = [item for item in batch_items if "image" not in item]
            image_only = [item for item in batch_items if "image" in item]
            sub_batches = [sb for sb in (text_only, image_only) if sb]

            for sub_batch in sub_batches:
                inputs = prepare_batch(sub_batch, tokenizer, image_processor, model, device)
                outputs = model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                images=inputs["images"])
                cls_logits = extract_answer_logits_batched(
                    outputs.logits, inputs["labels"], label_token_ids, device)
                preds = cls_logits.argmax(dim=-1)

                all_logits.append(cls_logits.cpu())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(inputs["class_labels"].cpu().tolist())

        logits_cat = torch.cat(all_logits)
        preds_t = torch.tensor(all_preds)
        labels_t = torch.tensor(all_labels)

        correct = (preds_t == labels_t).float()
        acc = correct.mean().item()

        per_class = {}
        for c in range(NUM_CLASSES):
            mask = labels_t == c
            if mask.sum() > 0:
                per_class[LABEL_CHARS[c]] = correct[mask].mean().item()

        valid_class_accs = [v for v in per_class.values()]
        worst_class_acc = min(valid_class_accs) if valid_class_accs else 0.0

        vwr = compute_vwr_gamma(logits_cat, labels_t, gamma, kappa, NUM_CLASSES)
        smax = compute_sigma_max(logits_cat, labels_t, gamma, kappa, NUM_CLASSES)

        margins = compute_margins(logits_cat, labels_t)
        gates = compute_gates(margins, gamma, kappa)
        fragile_ratio = (gates > 0.5).float().mean().item()
        avg_gate = gates.mean().item()

        results[etype] = {
            "eval_type": etype,
            "accuracy": acc,
            "worst_class_acc": worst_class_acc,
            "worst_class_err": 1.0 - worst_class_acc,
            "vwr_gamma": vwr,
            "sigma_max": smax,
            "per_class_acc": per_class,
            "avg_gate": avg_gate,
            "fragile_ratio": fragile_ratio,
            "n_samples": len(labels_t),
            "gamma": gamma,
            "kappa": kappa,
        }

        print(f"\n  [{etype.upper()}] Acc={acc:.4f}  Worst={worst_class_acc:.4f}  "
              f"VWR={vwr:.4f}  σ_max={smax:.4f}  "
              f"gate_avg={avg_gate:.4f}  fragile={fragile_ratio:.4f}")
        for cls_name, cls_acc in per_class.items():
            print(f"    Class {cls_name}: {cls_acc:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--eval_types", type=str, default="clean,sa,pa")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    eval_types = args.eval_types.split(",")
    results = evaluate(args.ckpt_dir, args.gamma, args.gpu,
                      eval_types=eval_types, kappa=args.kappa)

    out_path = args.output or os.path.join(
        os.path.dirname(args.ckpt_dir), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
