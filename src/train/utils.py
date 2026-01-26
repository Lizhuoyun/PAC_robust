import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list] = None) -> Dict[str, Any]:
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item}")
        key, raw = item.split("=", 1)
        keys = key.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        value: Any = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        elif raw.strip().startswith(("[", "{")):
            # Allow passing structured values (lists/dicts) from CLI, e.g.
            # --override lora.target_modules='["q_proj","k_proj","v_proj","o_proj"]'
            try:
                value = json.loads(raw)
            except Exception:
                value = raw
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        cur[keys[-1]] = value
    return cfg


def resolve_presets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for section in ("augment", "r3f", "smart"):
        if section not in cfg:
            continue
        preset = cfg[section].get("preset")
        presets = cfg[section].get("presets", {})
        if preset and preset in presets:
            resolved = dict(cfg[section])
            resolved.update(presets[preset])
            cfg[section] = resolved
    return cfg


def resolve_torch_dtype(cfg: Dict[str, Any]) -> Optional[torch.dtype]:
    """
    Optional torch dtype resolver from config.
    Supports:
      - cfg.model.torch_dtype: "bf16" | "fp16" | "fp32"
      - cfg.torch_dtype: same strings (fallback)
    """
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    raw = model_cfg.get("torch_dtype", cfg.get("torch_dtype"))
    if raw is None:
        return None
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp16", "float16", "half"):
            return torch.float16
        if key in ("fp32", "float32"):
            return torch.float32
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class JsonlLogger:
    path: str

    def log(self, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def maybe_init_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    if not cfg.get("wandb", {}).get("enabled", False):
        return None
    import wandb

    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "pac_robust"),
        name=wandb_cfg.get("name"),
        config=cfg,
    )
    return wandb


def timer() -> float:
    return time.time()


def elapsed(start: float) -> float:
    return time.time() - start
