"""
Modular perturbation functions for text and image inputs.
Each function takes an input and returns a perturbed copy.
"""
import random, string, io, copy
from PIL import Image, ImageFilter
import numpy as np


# ---------------------------------------------------------------------------
# Text perturbations
# ---------------------------------------------------------------------------

KEYBOARD_NEIGHBORS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfce', 'e': 'wrd',
    'f': 'dgrt', 'g': 'fhty', 'h': 'gjyu', 'i': 'uok', 'j': 'hkui',
    'k': 'jlio', 'l': 'kop', 'm': 'njk', 'n': 'bmhj', 'o': 'iplk',
    'p': 'ol', 'q': 'wa', 'r': 'eft', 's': 'adwx', 't': 'rgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'thu',
    'z': 'asx',
}

DISTRACTOR_SENTENCES = [
    "The weather today is sunny with a high of 75 degrees.",
    "Please remember to subscribe to our newsletter for updates.",
    "This information was last updated in January 2024.",
    "For more details, visit our website at example.com.",
    "Note: results may vary depending on individual circumstances.",
    "According to recent surveys, satisfaction rates have increased.",
    "The committee will meet again next Thursday to discuss further.",
    "Free shipping is available on orders over fifty dollars.",
    "Our offices are located in downtown metropolitan area.",
    "This product comes with a one-year limited warranty.",
]


def typo_perturbation(text: str, rate: float = 0.10, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    chars = list(text)
    n_changes = max(1, int(len(chars) * rate))
    positions = rng.sample(range(len(chars)), min(n_changes, len(chars)))
    for pos in positions:
        c = chars[pos].lower()
        action = rng.choice(["swap", "delete", "insert", "neighbor"])
        if action == "swap" and pos + 1 < len(chars):
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif action == "delete":
            chars[pos] = ""
        elif action == "insert":
            chars.insert(pos, rng.choice(string.ascii_lowercase))
        elif action == "neighbor" and c in KEYBOARD_NEIGHBORS:
            replacement = rng.choice(KEYBOARD_NEIGHBORS[c])
            chars[pos] = replacement if chars[pos].islower() else replacement.upper()
    return "".join(chars)


def distractor_perturbation(text: str, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    distractor = rng.choice(DISTRACTOR_SENTENCES)
    sentences = text.split(". ")
    if len(sentences) > 1:
        pos = rng.randint(1, len(sentences) - 1)
        sentences.insert(pos, distractor)
        return ". ".join(sentences)
    return text + " " + distractor


# --- Format-rewrite templates (per task) -----------------------------------

_AGNEWS_TEMPLATES = [
    "Read the following text and decide which category it belongs to.\nOptions: A) World  B) Sports  C) Business  D) Sci/Tech\n\nText: {text}\n\nYour answer:",
    "News classification task. Categories: (A) World, (B) Sports, (C) Business, (D) Sci/Tech.\n\n{text}\n\nCategory:",
    "Determine the topic of this article from [A. World, B. Sports, C. Business, D. Sci/Tech].\n\nArticle: {text}\n\nTopic:",
]

_ARC_TEMPLATES = [
    "Science question. Pick the best answer.\n\n{question}\n{choices}\n\nBest answer:",
    "Select the correct option for the following question.\n\nQ: {question}\n{choices}\n\nCorrect option:",
    "Answer this science question by choosing one letter.\n\n{question}\n{choices}\n\nLetter:",
]

_SQA_TEMPLATES = [
    "Look at the image and answer the question.\n\n{question}\n{choices}\n\nYour answer:",
    "Based on what you see, select the best option.\n\nQuestion: {question}\n{choices}\n\nOption:",
]

_ROBOT_TEMPLATES = [
    "Given the scene, choose the correct robot action.\n\nInstruction: {instruction}\n{choices}\n\nAction:",
    "Select the safest action for the robot.\n\nCommand: {instruction}\n{choices}\n\nChosen action:",
]

FORMAT_TEMPLATES = {
    "agnews": _AGNEWS_TEMPLATES,
    "arc": _ARC_TEMPLATES,
    "scienceqa": _SQA_TEMPLATES,
    "robot": _ROBOT_TEMPLATES,
}


def format_rewrite_perturbation(original_prompt: str, task_name: str,
                                 fill_fields: dict, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    templates = FORMAT_TEMPLATES.get(task_name, [])
    if not templates:
        return original_prompt
    tmpl = rng.choice(templates)
    try:
        return tmpl.format(**fill_fields)
    except KeyError:
        return original_prompt


# ---------------------------------------------------------------------------
# Image perturbations
# ---------------------------------------------------------------------------

def blur_perturbation(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=kernel_size))


def jpeg_perturbation(image: Image.Image, quality: int = 20) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def resize_perturbation(image: Image.Image, scale: float = 0.25) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def occlusion_perturbation(image: Image.Image, patch_ratio: float = 0.15,
                            rng: random.Random = None) -> Image.Image:
    if rng is None:
        rng = random.Random()
    img = image.copy()
    w, h = img.size
    pw, ph = int(w * patch_ratio), int(h * patch_ratio)
    x0, y0 = rng.randint(0, max(0, w - pw)), rng.randint(0, max(0, h - ph))
    arr = np.array(img)
    arr[y0:y0 + ph, x0:x0 + pw] = 128
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def apply_text_perturbation(text: str, ptype: str, task_name: str = "",
                            fill_fields: dict = None, cfg: dict = None,
                            rng: random.Random = None) -> str:
    rate = (cfg or {}).get("typo_rate", 0.10)
    if ptype == "typo":
        return typo_perturbation(text, rate=rate, rng=rng)
    elif ptype == "distractor":
        return distractor_perturbation(text, rng=rng)
    elif ptype == "format_rewrite":
        return format_rewrite_perturbation(text, task_name, fill_fields or {}, rng=rng)
    return text


def apply_image_perturbation(image: Image.Image, ptype: str, cfg: dict = None,
                              rng: random.Random = None) -> Image.Image:
    if image is None:
        return None
    cfg = cfg or {}
    if ptype == "blur":
        return blur_perturbation(image, kernel_size=cfg.get("blur_kernel", 5))
    elif ptype == "jpeg":
        return jpeg_perturbation(image, quality=cfg.get("jpeg_quality", 20))
    elif ptype == "resize":
        return resize_perturbation(image, scale=cfg.get("resize_scale", 0.25))
    elif ptype == "occlusion":
        return occlusion_perturbation(image, rng=rng)
    return image


def apply_perturbation(text: str, image, ptype: str, task_name: str = "",
                       fill_fields: dict = None, cfg: dict = None,
                       rng: random.Random = None):
    """Apply a named perturbation to text and/or image."""
    if rng is None:
        rng = random.Random()
    cfg = cfg or {}

    if ptype in ("typo", "distractor", "format_rewrite"):
        return apply_text_perturbation(text, ptype, task_name, fill_fields, cfg, rng), image
    elif ptype in ("blur", "jpeg", "resize", "occlusion"):
        return text, apply_image_perturbation(image, ptype, cfg, rng)
    elif ptype == "joint":
        text_p = rng.choice(["typo", "distractor"])
        img_p = rng.choice(["blur", "jpeg", "resize"])
        text = apply_text_perturbation(text, text_p, task_name, fill_fields, cfg, rng)
        image = apply_image_perturbation(image, img_p, cfg, rng)
        return text, image
    return text, image


def random_perturbation(text: str, image, task_name: str, modality: str,
                        fill_fields: dict = None, cfg: dict = None,
                        rng: random.Random = None):
    """Pick a random perturbation appropriate for the modality."""
    if rng is None:
        rng = random.Random()
    if modality == "text":
        ptypes = cfg.get("text_perturbation_types", ["typo", "distractor", "format_rewrite"])
    else:
        ptypes = (cfg.get("text_perturbation_types", ["typo", "distractor"]) +
                  cfg.get("image_perturbation_types", ["blur", "jpeg", "resize"]) +
                  ["joint"])
    ptype = rng.choice(ptypes)
    return apply_perturbation(text, image, ptype, task_name, fill_fields, cfg, rng)
