"""
Text perturbation functions for ARC-Challenge.

Three perturbation types:
  typo           — character-level noise (random delete/insert/swap/replace)
  distractor     — prepend an unrelated sentence
  format_rewrite — rewrite choice labels from letters to numbers/ordinals
"""
import random
import string

# ── Distractor sentence pool ───────────────────────────────────────────────

_DISTRACTORS = [
    "The weather today is partly cloudy with a chance of rain.",
    "Scientists recently discovered a new species of deep-sea fish.",
    "The stock market closed higher on Tuesday.",
    "A popular café downtown is offering a new seasonal menu.",
    "Engineers are developing faster battery charging technologies.",
    "The local library will host a book fair next weekend.",
    "Astronomers detected a faint signal from a distant galaxy.",
    "A team of archaeologists unearthed ancient pottery in Egypt.",
    "Researchers published new findings on sleep and memory.",
    "The city council approved a budget for park renovations.",
    "A marathon was held in the city attracting thousands of runners.",
    "New regulations on carbon emissions were announced by the agency.",
    "The university is expanding its online course offerings.",
    "Farmers reported an unusually large harvest this autumn.",
    "A documentary about ocean conservation won a major award.",
]


# ── Format-rewrite templates ───────────────────────────────────────────────

def _format_rewrite(question: str, choices: list, rng: random.Random) -> str:
    """Rewrite multiple-choice options using a different labelling style."""
    style = rng.choice(["numeric", "ordinal", "roman"])
    if style == "numeric":
        labels = [f"({i+1})" for i in range(len(choices))]
    elif style == "ordinal":
        labels = ["First", "Second", "Third", "Fourth"][: len(choices)]
    else:
        labels = ["I.", "II.", "III.", "IV."][: len(choices)]

    choice_str = "\n".join(f"{lbl} {c}" for lbl, c in zip(labels, choices))
    return (
        "Answer the following science question.\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        "Answer (use the original letter A/B/C/D):"
    )


# ── Typo perturbation ──────────────────────────────────────────────────────

def _typo_word(word: str, rng: random.Random) -> str:
    if len(word) <= 1:
        return word
    op = rng.choice(["delete", "insert", "swap", "replace"])
    i = rng.randint(0, len(word) - 1)
    if op == "delete":
        return word[:i] + word[i+1:]
    elif op == "insert":
        c = rng.choice(string.ascii_lowercase)
        return word[:i] + c + word[i:]
    elif op == "swap" and len(word) >= 2:
        j = rng.randint(0, len(word) - 2)
        w = list(word)
        w[j], w[j+1] = w[j+1], w[j]
        return "".join(w)
    else:
        c = rng.choice(string.ascii_lowercase)
        return word[:i] + c + word[i+1:]


def typo_perturbation(text: str, rate: float = 0.10,
                      rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    words = text.split()
    out = []
    for w in words:
        if rng.random() < rate:
            out.append(_typo_word(w, rng))
        else:
            out.append(w)
    return " ".join(out)


# ── Distractor perturbation ────────────────────────────────────────────────

def distractor_perturbation(text: str, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    distractor = rng.choice(_DISTRACTORS)
    return distractor + " " + text


# ── Main dispatch ──────────────────────────────────────────────────────────

def apply_perturbation(ex: dict, ptype: str, cfg: dict,
                       rng: random.Random = None) -> dict:
    """
    Apply a named perturbation to an ARC example.
    Returns a shallow copy with 'text' and 'prompt' replaced.
    ptype: 'clean' | 'typo' | 'distractor' | 'format_rewrite'
    """
    if rng is None:
        rng = random.Random()
    if ptype == "clean":
        return ex

    ex2 = dict(ex)  # shallow copy
    if ptype == "typo":
        ex2["text"]   = typo_perturbation(ex["text"], cfg["typo_rate"], rng)
        ex2["prompt"] = _rebuild_prompt(ex2)
    elif ptype == "distractor":
        ex2["text"]   = distractor_perturbation(ex["text"], rng)
        ex2["prompt"] = _rebuild_prompt(ex2)
    elif ptype == "format_rewrite":
        ex2["prompt"] = _format_rewrite(ex["text"], ex["choices"], rng)
        # text unchanged — only the prompt format changes
    return ex2


def _rebuild_prompt(ex: dict) -> str:
    from data import arc_prompt
    return arc_prompt(ex["text"], ex["choices"])


def random_perturbation(ex: dict, cfg: dict,
                        rng: random.Random = None) -> dict:
    """Sample one perturbation type uniformly at random and apply it."""
    if rng is None:
        rng = random.Random()
    ptype = rng.choice(cfg["perturb_types"])
    return apply_perturbation(ex, ptype, cfg, rng)
