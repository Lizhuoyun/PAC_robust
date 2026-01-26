from typing import Dict

import torch
import torch.nn.functional as F


def classification_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def lm_nll(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=ignore_index)


def mix_losses(clean: torch.Tensor, pert: torch.Tensor, mix_weight: float) -> torch.Tensor:
    return (1.0 - mix_weight) * clean + mix_weight * pert


def build_loss_dict(task_loss: torch.Tensor, **extras: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {"task": task_loss}
    out.update(extras)
    return out
