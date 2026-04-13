"""
Gamma calibration for ScienceQA: compute margin distribution from LoRA checkpoint
on SA-perturbed validation data, output quantile-based gamma values.

Usage:
  python sqa_gamma_calibrate.py --ckpt_dir <path> --gpu 0
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/LOCAL2/zhuoyun/PAC_robust/ARDS")
sys.path.insert(0, "/LOCAL2/zhuoyun/PAC_robust/new")

os.environ["HF_HOME"] = "/LOCAL2/zhuoyun/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/LOCAL2/zhuoyun/hf_cache"

from plugin import compute_margins

MODEL_PATH = "/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b"
LABEL_TOKEN_IDS = [319, 350, 315, 360]  # A, B, C, D
SA_TOKEN_MAP = {319: 660, 350: 399, 315: 382, 360: 390}
NUM_CLASSES = 4
IGNORE_INDEX = -100


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


def load_sqa_data(data_path):
    data = json.load(open(data_path))
    label_chars = ["A", "B", "C", "D"]
    for d in data:
        ans = d["conversations"][1]["value"].strip()
        if "The answer is " in ans:
            letter = ans.split("The answer is ")[-1].rstrip(".")
        else:
            letter = ans.strip().rstrip(".")
        d["label"] = label_chars.index(letter) if letter in label_chars else 0
    return data


def prepare_single_sample(item, tokenizer, image_processor, model, device,
                          apply_sa=False):
    """Prepare a single sample, optionally with SA perturbation."""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from PIL import Image

    conv = conv_templates["vicuna_v1"].copy()
    human_msg = item["conversations"][0]["value"]
    gpt_msg = item["conversations"][1]["value"]

    if apply_sa:
        import re
        sa_map = {"A": "Q", "B": "W", "C": "E", "D": "R"}
        for orig, repl in sa_map.items():
            human_msg = re.sub(rf'\(({orig})\)', f'({repl})', human_msg)
            human_msg = re.sub(rf'\b{orig}\.\s', f'{repl}. ', human_msg)

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
        img_folder = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/images"
        img_path = os.path.join(img_folder, item["image"])
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

    return input_ids, labels, images


def extract_class_logits(logits, input_ids, device):
    """Extract class logits at the position just before the answer token.
    input_ids after model forward may contain expanded image tokens,
    so we search within the actual input_ids (which matches logits dim 1)."""
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
def calibrate(ckpt_dir, data_path, device_id=0, n_samples=500):
    device = f"cuda:{device_id}"
    model, tokenizer, image_processor = load_model_with_lora(ckpt_dir, device)

    data = load_sqa_data(data_path)
    import random
    random.seed(42)
    random.shuffle(data)
    val_data = data[:n_samples]

    all_margins = []
    for item in tqdm(val_data, desc="Computing margins on SA-perturbed data"):
        input_ids, labels, images = prepare_single_sample(
            item, tokenizer, image_processor, model, device, apply_sa=True)

        outputs = model(input_ids=input_ids, images=images)
        cls_logits = extract_class_logits(outputs.logits, input_ids, device)
        label_t = torch.tensor([item["label"]], device=device)
        margin = compute_margins(cls_logits, label_t)
        all_margins.append(margin.cpu().item())

    margins = np.array(all_margins)
    result = {}
    for q in [0.10, 0.25, 0.50, 0.75]:
        result[f"q{int(q*100):02d}"] = float(np.quantile(margins, q))
    result["mean"] = float(margins.mean())
    result["std"] = float(margins.std())
    result["min"] = float(margins.min())
    result["max"] = float(margins.max())
    result["n_samples"] = len(margins)

    print(f"\nMargin distribution ({len(margins)} samples):")
    print(f"  mean={result['mean']:.4f}, std={result['std']:.4f}")
    print(f"  min={result['min']:.4f}, max={result['max']:.4f}")
    for k in ["q10", "q25", "q50", "q75"]:
        print(f"  {k} = {result[k]:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_path", default="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/sqa_selection/scienceqa_selected_subset.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output", default="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/gamma_calibration.json")
    args = parser.parse_args()

    result = calibrate(args.ckpt_dir, args.data_path, args.gpu, args.n_samples)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")
