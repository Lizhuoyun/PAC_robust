"""
ARC-Challenge data loading and prompt construction.
"""
import random
from datasets import load_dataset

from config import HF_CACHE

# ── Prompt template ────────────────────────────────────────────────────────

def arc_prompt(question: str, choices: list) -> str:
    """Build standard multiple-choice prompt."""
    choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (
        "Answer the following science question by choosing the best option.\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        "Answer:"
    )


# ── Label mapping ──────────────────────────────────────────────────────────

_ANSWER_MAP = {"1": 0, "2": 1, "3": 2, "4": 3,
               "A": 0, "B": 1, "C": 2, "D": 3}


def _convert_row(row):
    choices = row["choices"]["text"]
    answer  = row["answerKey"].strip()
    label   = _ANSWER_MAP.get(answer)
    if label is None or label >= len(choices):
        return None
    # Pad / truncate to exactly 4 choices
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


def _balanced_subsample(items: list, n: int, rng: random.Random) -> list:
    """Sample n items with roughly balanced labels."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for x in items:
        buckets[x["label"]].append(x)
    per_class = max(1, n // 4)
    out = []
    for cls in range(4):
        bucket = buckets[cls]
        rng.shuffle(bucket)
        out.extend(bucket[:per_class])
    rng.shuffle(out)
    return out[:n]


def load_arc(cfg: dict, seed: int = 42) -> dict:
    """
    Load ARC-Challenge and return {"train": [...], "val": [...], "test": [...]}.
    Each item is a dict with keys: text, choices, prompt, label, label_char, fill_fields, task.
    """
    rng = random.Random(seed)
    ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=HF_CACHE)

    def convert_split(hf_split):
        items = [_convert_row(r) for r in hf_split]
        return [x for x in items if x is not None]

    raw_train = convert_split(ds["train"])
    raw_val   = convert_split(ds["validation"])
    raw_test  = convert_split(ds["test"])

    train = _balanced_subsample(raw_train, cfg["train_size"], rng)
    val   = _balanced_subsample(raw_val,   cfg["val_size"],   random.Random(seed + 1))
    test  = _balanced_subsample(raw_test,  cfg["test_size"],  random.Random(seed + 2))

    print(f"  ARC data: train={len(train)} val={len(val)} test={len(test)}")
    return dict(train=train, val=val, test=test)


def rebuild_prompt(ex: dict, perturbed_text: str) -> str:
    """Rebuild arc_prompt with perturbed question text."""
    return arc_prompt(perturbed_text, ex["choices"])
