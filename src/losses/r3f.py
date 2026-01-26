from typing import Optional

import torch
import torch.nn.functional as F


def r3f_kl_logits(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    detach_target: bool = True,
) -> torch.Tensor:
    if detach_target:
        clean_logits = clean_logits.detach()
    clean_logp = F.log_softmax(clean_logits, dim=-1)
    noisy_logp = F.log_softmax(noisy_logits, dim=-1)
    clean_p = clean_logp.exp()
    kl = (clean_p * (clean_logp - noisy_logp)).sum(dim=-1)
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp_min(1.0)
    return kl.mean()
