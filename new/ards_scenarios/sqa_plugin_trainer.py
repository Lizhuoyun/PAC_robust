"""
ScienceQA LoRA / Plugin-LoRA Trainer
Integrates gamma-aware spectral plugin into LLaVA-v1.5 LoRA training.

Usage:
  python sqa_plugin_trainer.py --mode lora --gpu 0
  python sqa_plugin_trainer.py --mode plugin --gamma_q 0.25 --gpu 1
"""
import os, sys, json, math, random, copy, argparse, time, re
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

ARDS_DIR = "/LOCAL2/zhuoyun/PAC_robust/ARDS"
SCENARIOS_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"
PLUGIN_DIR = "/LOCAL2/zhuoyun/PAC_robust/new"
sys.path.insert(0, ARDS_DIR)
sys.path.insert(0, PLUGIN_DIR)

from plugin import (compute_margins, compute_gates, build_transition_matrix,
                     r_spec, plugin_loss, compute_vwr_gamma, compute_sigma_max)

os.environ["HF_HOME"] = "/LOCAL2/zhuoyun/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/LOCAL2/zhuoyun/hf_cache"

MODEL_PATH = "/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b"
DATA_PATH = os.path.join(SCENARIOS_DIR, "sqa_selection/scienceqa_selected_subset.json")
IMAGE_FOLDER = os.path.join(ARDS_DIR, "playground/data/eval/scienceqa/images")
TEST_CLEAN = os.path.join(ARDS_DIR, "playground/data/eval/scienceqa/llava_test_CQM-A.json")
TEST_SA = os.path.join(ARDS_DIR, "playground/data/eval/scienceqa/llava_test_CQM-A_convertedABCDE-QWERT.json")

LABEL_CHARS = ["A", "B", "C", "D"]
NUM_CLASSES = 4
KAPPA = 0.5
DEFAULT_ALPHA = 0.1
BETA = 0.0

LORA_R = 128
LORA_ALPHA = 256
LR = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 2048

# ─── Model loading ────────────────────────────────────────────────────────────

def load_llava_with_lora(device_id=0):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name)

    model.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for p in model.base_model.model.model.mm_projector.parameters():
        p.requires_grad = True

    model = model.to(f"cuda:{device_id}")
    model.train()

    return model, tokenizer, image_processor


def get_label_token_ids(tokenizer):
    ids = []
    for c in LABEL_CHARS:
        toks = tokenizer.encode(c, add_special_tokens=False)
        ids.append(toks[-1])
    return ids


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sqa_train_data():
    data = json.load(open(DATA_PATH))
    for d in data:
        ans_text = d["conversations"][1]["value"].strip()
        if "The answer is " in ans_text:
            letter = ans_text.split("The answer is ")[-1].rstrip(".")
        else:
            letter = ans_text.strip().rstrip(".")
        d["label"] = LABEL_CHARS.index(letter) if letter in LABEL_CHARS else 0
    return data


def load_sqa_test_data(filepath):
    data = json.load(open(filepath))
    for d in data:
        ans_text = d["conversations"][1]["value"].strip()
        if "The answer is " in ans_text:
            letter = ans_text.split("The answer is ")[-1].rstrip(".")
        else:
            letter = ans_text.strip().rstrip(".")
        d["label"] = LABEL_CHARS.index(letter) if letter in LABEL_CHARS else 0
    return data


# ─── Forward helpers ──────────────────────────────────────────────────────────

def prepare_batch(batch, tokenizer, image_processor, model, device):
    """Prepare a batch for LLaVA forward pass, return input dict + label info."""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from PIL import Image

    all_input_ids = []
    all_labels = []
    all_images = []
    class_labels = []

    for item in batch:
        conv = conv_templates["vicuna_v1"].copy()
        human_msg = item["conversations"][0]["value"]
        gpt_msg = item["conversations"][1]["value"]

        if "<image>" in human_msg:
            human_msg_clean = human_msg.replace("<image>", "").strip()
            human_msg_formatted = DEFAULT_IMAGE_TOKEN + "\n" + human_msg_clean
        else:
            human_msg_formatted = human_msg

        conv.append_message(conv.roles[0], human_msg_formatted)
        conv.append_message(conv.roles[1], gpt_msg)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt')
        all_input_ids.append(input_ids)

        target = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(sep)
        if len(parts) >= 2:
            instruction_part = parts[0] + sep
            inst_ids = tokenizer_image_token(instruction_part, tokenizer,
                                             IMAGE_TOKEN_INDEX, return_tensors='pt')
            target[:len(inst_ids)] = -100
        all_labels.append(target)

        if "image" in item:
            img_path = os.path.join(IMAGE_FOLDER, item["image"])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img_tensor = process_images([img], image_processor, model.config)[0]
                all_images.append(img_tensor)
            else:
                all_images.append(None)
        else:
            all_images.append(None)

        class_labels.append(item["label"])

    max_len = min(max(ids.shape[0] for ids in all_input_ids), MAX_SEQ_LEN)
    padded_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(all_input_ids, all_labels)):
        seq_len = min(ids.shape[0], max_len)
        padded_ids[i, :seq_len] = ids[:seq_len]
        padded_labels[i, :seq_len] = labs[:seq_len]
        attn_mask[i, :seq_len] = 1

    images_tensor = None
    if any(img is not None for img in all_images):
        valid_imgs = [img for img in all_images if img is not None]
        if valid_imgs and all(img.shape == valid_imgs[0].shape for img in valid_imgs):
            images_tensor = torch.stack(valid_imgs).half().to(device)

    return {
        "input_ids": padded_ids.to(device),
        "labels": padded_labels.to(device),
        "attention_mask": attn_mask.to(device),
        "images": images_tensor,
        "class_labels": torch.tensor(class_labels, device=device),
    }


def extract_answer_logits(logits, labels, label_token_ids, device):
    """
    Extract verbalizer logits at the answer prediction position.
    For each sample, find the position of the answer token in labels
    (first non -100 token in the GPT response that matches A/B/C/D).
    """
    B, T, V = logits.shape
    label_ids_t = torch.tensor(label_token_ids, device=device)
    label_len = labels.shape[1]

    answer_logits_list = []
    for i in range(B):
        answer_pos = -1
        for t in range(min(T, label_len) - 1, -1, -1):
            if labels[i, t] != -100 and labels[i, t].item() in label_token_ids:
                answer_pos = t - 1
                break
        if answer_pos < 0:
            non_ignore = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(non_ignore) > 0:
                answer_pos = non_ignore[0].item() - 1
            else:
                answer_pos = T - 2

        answer_pos = max(0, min(answer_pos, T - 1))
        sample_logits = logits[i, answer_pos, label_ids_t]
        answer_logits_list.append(sample_logits)

    return torch.stack(answer_logits_list)


# ─── Perturbation for SA (Symbol Attack) ─────────────────────────────────────

SA_MAP = {"A": "Q", "B": "W", "C": "E", "D": "R"}
SA_MAP_INV = {v: k for k, v in SA_MAP.items()}

def apply_symbol_attack(batch_items):
    """Create SA-perturbed versions of batch items."""
    import re
    perturbed = []
    for item in batch_items:
        item_p = copy.deepcopy(item)
        q = item_p["conversations"][0]["value"]
        for orig, repl in SA_MAP.items():
            q = re.sub(rf'\(({orig})\)', f'({repl})', q)
            q = re.sub(rf'\b{orig}\.\s', f'{repl}. ', q)
        item_p["conversations"][0]["value"] = q

        ans = item_p["conversations"][1]["value"]
        for orig, repl in SA_MAP.items():
            ans = ans.replace(f"The answer is {orig}.", f"The answer is {repl}.")
            if ans.strip() == orig:
                ans = repl
        item_p["conversations"][1]["value"] = ans
        perturbed.append(item_p)
    return perturbed


def apply_position_attack(batch_items, rng):
    """Create PA-perturbed versions of batch items."""
    perturbed = []
    for item in batch_items:
        item_p = copy.deepcopy(item)
        q = item_p["conversations"][0]["value"]
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
            perturbed.append(item_p)
            continue

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

        item_p["conversations"][0]["value"] = "\n".join(lines)
        item_p["conversations"][1]["value"] = f"The answer is {LABEL_CHARS[new_label]}."
        item_p["label"] = new_label
        perturbed.append(item_p)
    return perturbed


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, tokenizer, image_processor, train_data, optimizer,
                    scheduler, label_token_ids, device, mode="lora",
                    gamma=None, alpha=DEFAULT_ALPHA, epoch=0):
    model.train()
    random.shuffle(train_data)
    total_loss = 0.0
    total_ce = 0.0
    total_rspec = 0.0
    n_steps = 0

    pbar = tqdm(range(0, len(train_data), BATCH_SIZE),
                desc=f"Epoch {epoch+1}")

    for step_idx, start in enumerate(pbar):
        batch_items = train_data[start:start + BATCH_SIZE]
        if not batch_items:
            continue

        inputs = prepare_batch(batch_items, tokenizer, image_processor, model, device)

        outputs = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"],
                       images=inputs["images"],
                       labels=inputs["labels"])

        ce_loss = outputs.loss
        class_logits = extract_answer_logits(
            outputs.logits, inputs["labels"], label_token_ids, device)
        class_labels = inputs["class_labels"]
        ce_value = ce_loss.item()

        if mode == "plugin" and gamma is not None:
            ce_term = ce_loss / GRAD_ACCUM
            ce_term.backward()

            clean_logits = class_logits.detach()
            del outputs, class_logits

            sa_items = apply_symbol_attack(batch_items)
            sa_inputs = prepare_batch(sa_items, tokenizer, image_processor, model, device)
            sa_outputs = model(input_ids=sa_inputs["input_ids"],
                              attention_mask=sa_inputs["attention_mask"],
                              images=sa_inputs["images"],
                              labels=sa_inputs["labels"])
            sa_logits = extract_answer_logits(
                sa_outputs.logits, sa_inputs["labels"], label_token_ids, device)

            rs, rst, reg = plugin_loss(
                clean_logits=clean_logits,
                perturbed_logits=sa_logits,
                labels=class_labels,
                gamma=gamma, kappa=KAPPA,
                num_classes=NUM_CLASSES,
                alpha=alpha, beta=BETA)

            reg_term = reg / GRAD_ACCUM
            reg_term.backward()
            loss_value = ce_value + reg.item()
            total_rspec += rs.item()
        else:
            loss = ce_loss / GRAD_ACCUM
            loss.backward()
            loss_value = ce_value

        if (step_idx + 1) % GRAD_ACCUM == 0 or start + BATCH_SIZE >= len(train_data):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            n_steps += 1

        total_loss += loss_value
        total_ce += ce_value

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            ce=f"{ce_value:.4f}",
            rspec=f"{total_rspec/(step_idx+1):.4f}" if mode == "plugin" else "N/A"
        )

    n_batches = max(1, len(range(0, len(train_data), BATCH_SIZE)))
    return total_loss / n_batches, total_ce / n_batches


def train_model(mode="lora", gamma=None, gamma_q_name="", device_id=0, seed=42,
                alpha=DEFAULT_ALPHA, output_suffix=None):
    torch.manual_seed(seed)
    random.seed(seed)
    device = f"cuda:{device_id}"

    suffix = output_suffix or (f"lora" if mode == "lora" else f"plugin_g{gamma_q_name}")
    ckpt_dir = os.path.join(SCENARIOS_DIR, "checkpoints", f"sqa_{suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training ScienceQA: mode={mode}, gamma={gamma}, alpha={alpha}, suffix={suffix}")
    print(f"Output: {ckpt_dir}")
    print(f"{'='*60}")

    model, tokenizer, image_processor = load_llava_with_lora(device_id)
    label_token_ids = get_label_token_ids(tokenizer)
    train_data = load_sqa_train_data()
    print(f"Train data: {len(train_data)} samples")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
    total_opt_steps = math.ceil(steps_per_epoch / GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = int(total_opt_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(NUM_EPOCHS):
        avg_loss, avg_ce = train_one_epoch(
            model, tokenizer, image_processor, train_data, optimizer, scheduler,
            label_token_ids, device, mode=mode, gamma=gamma, alpha=alpha, epoch=epoch)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, ce={avg_ce:.4f}")

    print(f"Saving to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return ckpt_dir


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(ckpt_dir, eval_type="clean", gamma=0.0, device_id=0):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from peft import PeftModel

    device = f"cuda:{device_id}"
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name)
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model = model.to(device)
    model.eval()

    label_token_ids = get_label_token_ids(tokenizer)

    rng = random.Random(42)

    if eval_type == "clean":
        test_data = load_sqa_test_data(TEST_CLEAN)
    elif eval_type == "sa":
        test_data = load_sqa_test_data(TEST_SA)
        for d in test_data:
            ans = d["conversations"][1]["value"].strip()
            for k, v in SA_MAP_INV.items():
                if f"The answer is {k}" in ans:
                    d["label"] = LABEL_CHARS.index(v)
                    break
    elif eval_type == "pa":
        test_data = load_sqa_test_data(TEST_CLEAN)
    else:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    all_logits = []
    all_labels = []
    all_preds = []

    for start in tqdm(range(0, len(test_data), BATCH_SIZE),
                      desc=f"Eval {eval_type}", leave=False):
        batch_items = test_data[start:start + BATCH_SIZE]
        if eval_type == "pa":
            batch_items = apply_position_attack(batch_items, rng)

        text_only = [item for item in batch_items if "image" not in item]
        image_only = [item for item in batch_items if "image" in item]
        sub_batches = [sb for sb in (text_only, image_only) if sb]

        for sub_batch in sub_batches:
            inputs = prepare_batch(sub_batch, tokenizer, image_processor, model, device)

            outputs = model(input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           images=inputs["images"])

            class_logits = extract_answer_logits(
                outputs.logits, inputs["labels"], label_token_ids, device)

            preds = class_logits.argmax(dim=-1)
            all_logits.append(class_logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(inputs["class_labels"].cpu())

    logits_cat = torch.cat(all_logits)
    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)

    correct = (preds_cat == labels_cat).float()
    acc = correct.mean().item()

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels_cat == c
        if mask.sum() > 0:
            per_class[LABEL_CHARS[c]] = correct[mask].mean().item()

    worst_class_acc = min(per_class.values()) if per_class else 0.0
    worst_class_err = 1.0 - worst_class_acc

    vwr = compute_vwr_gamma(logits_cat, labels_cat, gamma, KAPPA, NUM_CLASSES)
    smax = compute_sigma_max(logits_cat, labels_cat, gamma, KAPPA, NUM_CLASSES)

    del model
    torch.cuda.empty_cache()

    return {
        "eval_type": eval_type,
        "accuracy": acc,
        "worst_class_acc": worst_class_acc,
        "worst_class_err": worst_class_err,
        "vwr_gamma": vwr,
        "sigma_max": smax,
        "per_class_acc": per_class,
        "n_samples": len(labels_cat),
        "gamma_used": gamma,
    }


# ─── Gamma calibration ───────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_gamma_from_data(ckpt_dir, device_id=0, n_samples=500):
    """Calibrate gamma from margin distribution on SA-perturbed val data."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from peft import PeftModel

    device = f"cuda:{device_id}"
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name)
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model = model.to(device)
    model.eval()

    label_token_ids = get_label_token_ids(tokenizer)
    train_data = load_sqa_train_data()
    random.seed(42)
    random.shuffle(train_data)
    val_subset = train_data[:n_samples]

    sa_items = apply_symbol_attack(val_subset)

    all_margins = []
    for start in tqdm(range(0, len(sa_items), BATCH_SIZE), desc="Gamma calibration"):
        batch = sa_items[start:start + BATCH_SIZE]
        for i, item in enumerate(batch):
            item["label"] = val_subset[start + i]["label"]

        inputs = prepare_batch(batch, tokenizer, image_processor, model, device)
        outputs = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"],
                       images=inputs["images"])

        class_logits = extract_answer_logits(
            outputs.logits, inputs["labels"], label_token_ids, device)
        margins = compute_margins(class_logits, inputs["class_labels"])
        all_margins.append(margins.cpu())

    margins_all = torch.cat(all_margins).numpy()

    result = {}
    for q in [0.10, 0.25, 0.50]:
        result[f"q{int(q*100):02d}"] = float(np.quantile(margins_all, q))
    result["mean"] = float(margins_all.mean())
    result["std"] = float(margins_all.std())
    result["min"] = float(margins_all.min())
    result["max"] = float(margins_all.max())

    print(f"Margin distribution: mean={result['mean']:.4f}, std={result['std']:.4f}")
    for k, v in result.items():
        if k.startswith("q"):
            print(f"  {k} = {v:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BATCH_SIZE, GRAD_ACCUM
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lora", "plugin", "calibrate", "eval", "all"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gamma_q", type=str, default="q25",
                        help="Gamma quantile name (q10/q25/q50) or float value")
    parser.add_argument("--gamma", type=float, default=None, help="Explicit gamma value")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Plugin spectral loss weight")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--output_suffix", type=str, default=None, help="Optional checkpoint suffix")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_type", type=str, default="clean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.grad_accum is not None:
        GRAD_ACCUM = args.grad_accum

    if args.mode == "lora":
        ckpt = train_model(mode="lora", device_id=args.gpu, seed=args.seed)
        print(f"LoRA training done: {ckpt}")

    elif args.mode == "calibrate":
        ckpt = args.ckpt or os.path.join(SCENARIOS_DIR, "checkpoints/sqa_lora")
        result = calibrate_gamma_from_data(ckpt, device_id=args.gpu)
        out_path = os.path.join(SCENARIOS_DIR, "gamma_calibration.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved gamma calibration to {out_path}")

    elif args.mode == "plugin":
        gamma_val = args.gamma
        if gamma_val is None:
            cal_path = os.path.join(SCENARIOS_DIR, "gamma_calibration.json")
            if os.path.exists(cal_path):
                cal = json.load(open(cal_path))
                gamma_val = cal.get(args.gamma_q, 0.1)
            else:
                print("No gamma calibration found. Run --mode calibrate first.")
                return
        ckpt = train_model(mode="plugin", gamma=gamma_val,
                          gamma_q_name=args.gamma_q, device_id=args.gpu, seed=args.seed,
                          alpha=args.alpha, output_suffix=args.output_suffix)
        print(f"Plugin training done: {ckpt}")

    elif args.mode == "eval":
        ckpt = args.ckpt
        if not ckpt:
            print("Provide --ckpt for evaluation")
            return
        gamma_val = args.gamma or 0.1
        result = evaluate_checkpoint(ckpt, eval_type=args.eval_type,
                                    gamma=gamma_val, device_id=args.gpu)
        print(json.dumps({k: v for k, v in result.items()
                         if k != "per_class_acc"}, indent=2))
        print(f"Per-class: {result['per_class_acc']}")

    elif args.mode == "all":
        print("Running full pipeline: LoRA -> calibrate -> Plugin q10/q25/q50 -> eval all")
        lora_ckpt = train_model(mode="lora", device_id=args.gpu, seed=args.seed)

        cal = calibrate_gamma_from_data(lora_ckpt, device_id=args.gpu)
        cal_path = os.path.join(SCENARIOS_DIR, "gamma_calibration.json")
        with open(cal_path, "w") as f:
            json.dump(cal, f, indent=2)

        plugin_ckpts = {}
        for qname in ["q10", "q25", "q50"]:
            gamma_val = cal[qname]
            ckpt = train_model(mode="plugin", gamma=gamma_val,
                              gamma_q_name=qname, device_id=args.gpu, seed=args.seed)
            plugin_ckpts[qname] = ckpt

        all_results = []
        for method, ckpt in [("lora", lora_ckpt)] + [(f"plugin_{q}", c) for q, c in plugin_ckpts.items()]:
            gamma_eval = cal.get("q25", 0.1) if method == "lora" else cal.get(method.split("_")[-1], 0.1)
            for etype in ["clean", "sa"]:
                r = evaluate_checkpoint(ckpt, eval_type=etype, gamma=gamma_eval, device_id=args.gpu)
                r["method"] = method
                r["gamma_train"] = 0.0 if method == "lora" else gamma_eval
                all_results.append(r)

        out_path = os.path.join(SCENARIOS_DIR, "plugin_experiment_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
