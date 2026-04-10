"""
Synthetic robot action-selection dataset.
Generates simple scene images + text instructions → discrete action labels.
"""
import random, math
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ACTION_NAMES = ["pick", "push", "place", "move_left", "move_right", "stop"]

OBJECT_COLORS = {
    "red_block": (220, 50, 50),
    "blue_block": (50, 50, 220),
    "green_block": (50, 200, 50),
    "yellow_ball": (220, 220, 50),
    "obstacle": (100, 100, 100),
    "target_zone": (50, 200, 150),
}

INSTRUCTION_TEMPLATES = {
    "pick":       ["Pick up the {obj}.", "Grab the {obj} from the table.", "Lift the {obj}."],
    "push":       ["Push the {obj} forward.", "Slide the {obj} to the right.", "Move the {obj} by pushing."],
    "place":      ["Place the {obj} on the target.", "Put down the {obj} in the green zone.", "Set the {obj} on the target area."],
    "move_left":  ["Move left to avoid the obstacle.", "Go left.", "Shift the arm to the left."],
    "move_right": ["Move right toward the {obj}.", "Go right.", "Shift to the right side."],
    "stop":       ["Stop. Do not move.", "Halt immediately.", "Freeze — obstacle detected."],
}


def _draw_scene(objects: list, width=224, height=224, rng=None) -> Image.Image:
    if rng is None:
        rng = random.Random()
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # table surface
    draw.rectangle([0, height // 2, width, height], fill=(180, 150, 120))

    for obj_name, (cx, cy) in objects:
        color = OBJECT_COLORS.get(obj_name, (128, 128, 128))
        if "ball" in obj_name:
            r = rng.randint(12, 20)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0))
        elif "target" in obj_name:
            draw.rectangle([cx - 25, cy - 25, cx + 25, cy + 25],
                           fill=color, outline=(0, 0, 0), width=2)
        elif "obstacle" in obj_name:
            draw.polygon([(cx, cy - 20), (cx - 18, cy + 15), (cx + 18, cy + 15)],
                         fill=color, outline=(0, 0, 0))
        else:
            s = rng.randint(15, 22)
            draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color, outline=(0, 0, 0))
    return img


def _generate_example(label: int, rng: random.Random) -> dict:
    action = ACTION_NAMES[label]

    object_pool = list(OBJECT_COLORS.keys())
    scene_objs = []
    n_objs = rng.randint(2, 4)
    chosen = rng.sample(object_pool, min(n_objs, len(object_pool)))
    positions = []
    for obj_name in chosen:
        cx = rng.randint(30, 194)
        cy = rng.randint(50, 200)
        scene_objs.append((obj_name, (cx, cy)))
        positions.append((cx, cy))

    obj_for_text = rng.choice(chosen) if chosen else "object"
    obj_display = obj_for_text.replace("_", " ")

    templates = INSTRUCTION_TEMPLATES.get(action, ["Do something."])
    instruction = rng.choice(templates)
    try:
        instruction = instruction.format(obj=obj_display)
    except KeyError:
        pass

    image = _draw_scene(scene_objs, rng=rng)

    return dict(
        text=instruction,
        prompt=None,  # built later
        label=label,
        label_char=chr(65 + label),
        image=image,
        task="robot",
        fill_fields=dict(instruction=instruction),
    )


def generate_robot_dataset(cfg: dict, seed: int = 42) -> dict:
    from data_utils import _robot_prompt
    rng = random.Random(seed)
    num_classes = cfg["num_classes"]
    action_names = cfg["label_names"]

    def make_split(n):
        per_class = max(1, n // num_classes)
        examples = []
        for cls in range(num_classes):
            for _ in range(per_class):
                ex = _generate_example(cls, rng)
                ex["prompt"] = _robot_prompt(ex["text"], action_names)
                examples.append(ex)
        rng.shuffle(examples)
        return examples[:n]

    return dict(
        train=make_split(cfg["train_size"]),
        val=make_split(cfg["val_size"]),
        test=make_split(cfg["test_size"]),
    )
