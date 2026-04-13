"""
Dataset loading and prompt construction for ARC-Challenge and BoolQ.
"""
import random
from collections import defaultdict
from datasets import load_dataset

from config import HF_CACHE


# ═══════════════════════════════════════════════════════════════════════════
#  ARC-Challenge
# ═══════════════════════════════════════════════════════════════════════════

def arc_prompt(question: str, choices: list) -> str:
    choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (
        "Answer the following science question by choosing the best option.\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        "Answer:"
    )


_ARC_ANSWER_MAP = {"1": 0, "2": 1, "3": 2, "4": 3,
                   "A": 0, "B": 1, "C": 2, "D": 3}


def _convert_arc_row(row):
    choices = row["choices"]["text"]
    answer  = row["answerKey"].strip()
    label   = _ARC_ANSWER_MAP.get(answer)
    if label is None or label >= len(choices):
        return None
    choices = (choices + ["[N/A]"] * 4)[:4]
    q = row["question"].strip()
    return dict(
        text        = q,
        choices     = choices,
        prompt      = arc_prompt(q, choices),
        label       = label,
        label_char  = chr(65 + label),
        fill_fields = dict(question=q,
                           choices="\n".join(f"{chr(65+i)}. {c}"
                                             for i, c in enumerate(choices))),
        task        = "arc",
    )


def _balanced_subsample(items, n, num_classes, rng):
    buckets = defaultdict(list)
    for x in items:
        buckets[x["label"]].append(x)
    per_class = max(1, n // num_classes)
    out = []
    for cls in range(num_classes):
        bucket = buckets[cls]
        rng.shuffle(bucket)
        out.extend(bucket[:per_class])
    rng.shuffle(out)
    return out[:n]


def load_arc(cfg, seed=42):
    rng = random.Random(seed)
    ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=HF_CACHE)

    def convert_split(hf_split):
        items = [_convert_arc_row(r) for r in hf_split]
        return [x for x in items if x is not None]

    raw_train = convert_split(ds["train"])
    raw_val   = convert_split(ds["validation"])
    raw_test  = convert_split(ds["test"])

    K = cfg["num_classes"]
    train = _balanced_subsample(raw_train, cfg["train_size"], K, rng)
    val   = _balanced_subsample(raw_val,   cfg["val_size"],   K, random.Random(seed + 1))
    test  = _balanced_subsample(raw_test,  cfg["test_size"],  K, random.Random(seed + 2))

    print(f"  ARC data: train={len(train)} val={len(val)} test={len(test)}")
    return dict(train=train, val=val, test=test)


def rebuild_prompt(ex, perturbed_text):
    if ex.get("task") == "boolq":
        return boolq_prompt(ex["passage"], perturbed_text)
    return arc_prompt(perturbed_text, ex["choices"])


# ═══════════════════════════════════════════════════════════════════════════
#  BoolQ  (2 classes: A = False, B = True)
# ═══════════════════════════════════════════════════════════════════════════

def boolq_prompt(passage: str, question: str) -> str:
    return (
        "Read the passage and answer the question with A (No) or B (Yes).\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        "A. No\n"
        "B. Yes\n\n"
        "Answer:"
    )


def _convert_boolq_row(row):
    passage  = row["passage"].strip()
    question = row["question"].strip()
    answer   = row["answer"]          # bool: True / False
    label    = 1 if answer else 0     # A=No(0), B=Yes(1)
    return dict(
        text        = question,
        passage     = passage,
        choices     = ["No", "Yes"],
        prompt      = boolq_prompt(passage, question),
        label       = label,
        label_char  = "B" if answer else "A",
        fill_fields = dict(passage=passage, question=question),
        task        = "boolq",
    )


def load_boolq(cfg, seed=42):
    rng = random.Random(seed)
    ds  = load_dataset("google/boolq", cache_dir=HF_CACHE)

    raw_train = [_convert_boolq_row(r) for r in ds["train"]]
    raw_val   = [_convert_boolq_row(r) for r in ds["validation"]]

    K = cfg["num_classes"]
    train = _balanced_subsample(raw_train, cfg["train_size"], K, rng)
    # Split val into val + test
    rng_v = random.Random(seed + 1)
    rng_v.shuffle(raw_val)
    n_val  = cfg["val_size"]
    n_test = cfg["test_size"]
    val  = _balanced_subsample(raw_val[:n_val + n_test // 2], n_val, K, random.Random(seed + 1))
    test = _balanced_subsample(raw_val[n_val + n_test // 2:], n_test, K, random.Random(seed + 2))
    if len(test) < n_test:
        test = _balanced_subsample(raw_val, n_test, K, random.Random(seed + 3))

    print(f"  BoolQ data: train={len(train)} val={len(val)} test={len(test)}")
    return dict(train=train, val=val, test=test)


# ═══════════════════════════════════════════════════════════════════════════
#  Unified loader
# ═══════════════════════════════════════════════════════════════════════════

def load_data(cfg, seed=42):
    task = cfg["task_name"]
    if task == "arc":
        return load_arc(cfg, seed)
    elif task == "boolq":
        return load_boolq(cfg, seed)
    else:
        raise ValueError(f"Unknown task: {task}")
