"""
Text perturbation functions (task-agnostic).

Three perturbation types:
  typo           — character-level noise (random delete/insert/swap/replace)
  distractor     — prepend an unrelated sentence to the question text
  format_rewrite — rewrite prompt format (ARC: relabel choices; BoolQ: rephrase)
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


# ── Typo perturbation ────────────────────────────────────────────────────

def _typo_word(word, rng):
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


def typo_perturbation(text, rate=0.10, rng=None):
    if rng is None:
        rng = random.Random()
    words = text.split()
    out = [_typo_word(w, rng) if rng.random() < rate else w for w in words]
    return " ".join(out)


# ── Distractor perturbation ──────────────────────────────────────────────

def distractor_perturbation(text, rng=None):
    if rng is None:
        rng = random.Random()
    return rng.choice(_DISTRACTORS) + " " + text


# ── Format-rewrite (ARC) ─────────────────────────────────────────────────

def _format_rewrite_arc(question, choices, rng):
    style = rng.choice(["numeric", "ordinal", "roman"])
    if style == "numeric":
        labels = [f"({i+1})" for i in range(len(choices))]
    elif style == "ordinal":
        labels = ["First", "Second", "Third", "Fourth"][:len(choices)]
    else:
        labels = ["I.", "II.", "III.", "IV."][:len(choices)]
    choice_str = "\n".join(f"{lbl} {c}" for lbl, c in zip(labels, choices))
    return (
        "Answer the following science question.\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        "Answer (use the original letter A/B/C/D):"
    )


# ── Format-rewrite (BoolQ) ───────────────────────────────────────────────

def _format_rewrite_boolq(passage, question, rng):
    style = rng.choice(["yn", "tf", "agree"])
    if style == "yn":
        return (
            f"Based on the following text, answer Yes or No.\n\n"
            f"{passage}\n\n"
            f"Q: {question}\n\n"
            "A. No\nB. Yes\n\nAnswer:"
        )
    elif style == "tf":
        return (
            f"Read the text and decide if the statement is True or False.\n\n"
            f"Text: {passage}\n\n"
            f"Statement: {question}\n\n"
            "A. False\nB. True\n\nAnswer:"
        )
    else:
        return (
            f"Passage: {passage}\n\n"
            f"Do you agree with the following? {question}\n\n"
            "A. Disagree\nB. Agree\n\nAnswer:"
        )


# ── Main dispatch ────────────────────────────────────────────────────────

def apply_perturbation(ex, ptype, cfg, rng=None):
    if rng is None:
        rng = random.Random()
    if ptype == "clean":
        return ex

    ex2 = dict(ex)
    task = ex.get("task", cfg.get("task_name", "arc"))

    if ptype == "typo":
        ex2["text"] = typo_perturbation(ex["text"], cfg["typo_rate"], rng)
        ex2["prompt"] = _rebuild_prompt(ex2, task)
    elif ptype == "distractor":
        ex2["text"] = distractor_perturbation(ex["text"], rng)
        ex2["prompt"] = _rebuild_prompt(ex2, task)
    elif ptype == "format_rewrite":
        if task == "boolq":
            ex2["prompt"] = _format_rewrite_boolq(
                ex["passage"], ex["text"], rng)
        else:
            ex2["prompt"] = _format_rewrite_arc(
                ex["text"], ex["choices"], rng)
    return ex2


def _rebuild_prompt(ex, task):
    if task == "boolq":
        from data import boolq_prompt
        return boolq_prompt(ex["passage"], ex["text"])
    else:
        from data import arc_prompt
        return arc_prompt(ex["text"], ex["choices"])


def random_perturbation(ex, cfg, rng=None):
    if rng is None:
        rng = random.Random()
    ptype = rng.choice(cfg["perturb_types"])
    return apply_perturbation(ex, ptype, cfg, rng)
