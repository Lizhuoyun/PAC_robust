"""
Dataset loading and prompt formatting for all four tasks.
Returns lightweight dicts; tokenisation happens in the training loop.
"""
import os, random, json
from typing import List, Dict, Optional
from PIL import Image
from datasets import load_dataset

from config import HF_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (default — perturbation module may rewrite these)
# ──────────────────────────────────────────────────────────────────────────────

def _agnews_prompt(text: str) -> str:
    return (
        "Classify the following news article into one of the categories.\n"
        "A. World\nB. Sports\nC. Business\nD. Sci/Tech\n\n"
        f"Article: {text}\n\nAnswer:"
    )

def _arc_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (
        "Answer the following science question by selecting the correct option.\n\n"
        f"Question: {question}\n{opts}\n\nAnswer:"
    )

def _sqa_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (
        "Look at the image and answer the question by selecting the correct option.\n\n"
        f"Question: {question}\n{opts}\n\nAnswer:"
    )

def _robot_prompt(instruction: str, action_names: List[str]) -> str:
    opts = "\n".join(f"{chr(65+i)}. {a}" for i, a in enumerate(action_names))
    return (
        "Based on the scene and the instruction, select the most appropriate robot action.\n\n"
        f"Instruction: {instruction}\n{opts}\n\nAction:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _balanced_subsample(examples: list, label_key: str, n: int,
                        rng: random.Random) -> list:
    from collections import defaultdict
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex[label_key]].append(ex)
    per_class = max(1, n // len(by_class))
    out = []
    for cls, items in by_class.items():
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out[:n]


# ──────────────────────────────────────────────────────────────────────────────
# AG News
# ──────────────────────────────────────────────────────────────────────────────

def load_agnews(cfg: dict, seed: int = 42) -> dict:
    rng = random.Random(seed)
    ds = load_dataset("ag_news", cache_dir=HF_CACHE)
    label_chars = cfg["label_chars"]
    label_names = cfg["label_names"]

    def convert(row):
        label = row["label"]
        text = row["text"]
        return dict(
            text=text,
            prompt=_agnews_prompt(text),
            label=label,
            label_char=label_chars[label],
            image=None,
            task="agnews",
            fill_fields=dict(text=text),
        )

    train_all = [convert(r) for r in ds["train"]]
    test_all = [convert(r) for r in ds["test"]]

    train = _balanced_subsample(train_all, "label", cfg["train_size"], rng)
    rng2 = random.Random(seed + 1)
    rest = _balanced_subsample(test_all, "label",
                               cfg["val_size"] + cfg["test_size"], rng2)
    val = rest[:cfg["val_size"]]
    test = rest[cfg["val_size"]:cfg["val_size"] + cfg["test_size"]]

    return dict(train=train, val=val, test=test)


# ──────────────────────────────────────────────────────────────────────────────
# ARC-Challenge
# ──────────────────────────────────────────────────────────────────────────────

def load_arc(cfg: dict, seed: int = 42) -> dict:
    rng = random.Random(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=HF_CACHE)

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}

    def convert(row):
        choices = row["choices"]["text"]
        labels_raw = row["choices"]["label"]
        if len(choices) != 4:
            return None
        answer_key = row["answerKey"]
        label = label_map.get(answer_key)
        if label is None:
            return None
        question = row["question"]
        return dict(
            text=question,
            prompt=_arc_prompt(question, choices),
            label=label,
            label_char=cfg["label_chars"][label],
            image=None,
            task="arc",
            choices=choices,
            fill_fields=dict(
                question=question,
                choices="\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices)),
            ),
        )

    splits = {}
    for split_name, ds_split in [("train", ds["train"]), ("val", ds["validation"]),
                                  ("test", ds["test"])]:
        items = [convert(r) for r in ds_split]
        items = [x for x in items if x is not None]
        splits[split_name] = items

    train = _balanced_subsample(splits["train"], "label", cfg["train_size"], rng)
    val = _balanced_subsample(splits["val"], "label", cfg["val_size"], random.Random(seed+1))
    test = _balanced_subsample(splits["test"], "label", cfg["test_size"], random.Random(seed+2))
    return dict(train=train, val=val, test=test)


# ──────────────────────────────────────────────────────────────────────────────
# ScienceQA (image subset)
# ──────────────────────────────────────────────────────────────────────────────

def load_scienceqa(cfg: dict, seed: int = 42) -> dict:
    rng = random.Random(seed)
    try:
        ds = load_dataset("derek-thomas/ScienceQA", cache_dir=HF_CACHE)
    except Exception:
        ds = load_dataset("ScienceQA", cache_dir=HF_CACHE)

    def convert(row):
        if row.get("image") is None:
            return None
        choices = row["choices"]
        if len(choices) < 2 or len(choices) > 4:
            return None
        while len(choices) < 4:
            choices.append("[N/A]")
        label = row["answer"]
        if label >= 4:
            return None
        question = row["question"]
        image = row["image"]
        if not isinstance(image, Image.Image):
            return None
        return dict(
            text=question,
            prompt=_sqa_prompt(question, choices),
            label=label,
            label_char=cfg["label_chars"][label],
            image=image.convert("RGB"),
            task="scienceqa",
            choices=choices,
            fill_fields=dict(
                question=question,
                choices="\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices)),
            ),
        )

    splits_raw = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            items = [convert(r) for r in ds[split_name]]
            splits_raw[split_name] = [x for x in items if x is not None]

    if "validation" not in splits_raw:
        splits_raw["validation"] = splits_raw.get("test", [])

    train = _balanced_subsample(splits_raw.get("train", []), "label", cfg["train_size"], rng)
    val = _balanced_subsample(splits_raw.get("validation", []), "label", cfg["val_size"],
                              random.Random(seed + 1))
    test = _balanced_subsample(splits_raw.get("test", []), "label", cfg["test_size"],
                               random.Random(seed + 2))
    return dict(train=train, val=val, test=test)


# ──────────────────────────────────────────────────────────────────────────────
# Unified loader
# ──────────────────────────────────────────────────────────────────────────────

def load_task_data(cfg: dict, seed: int = 42) -> dict:
    task = cfg["task_name"]
    if task == "agnews":
        return load_agnews(cfg, seed)
    elif task == "arc":
        return load_arc(cfg, seed)
    elif task == "scienceqa":
        return load_scienceqa(cfg, seed)
    elif task == "robot":
        from robot_data import generate_robot_dataset
        return generate_robot_dataset(cfg, seed)
    else:
        raise ValueError(f"Unknown task: {task}")
