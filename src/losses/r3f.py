from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.losses.spectral import batch_transition_matrix


def _candidate_probs_from_logits(
    logits: torch.Tensor, gold_ids: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = logits.clone()
    masked[torch.arange(logits.size(0), device=logits.device), gold_ids] = -1e9
    topk = torch.topk(masked, k=top_k, dim=-1)
    cand_ids = torch.cat([gold_ids.unsqueeze(1), topk.indices], dim=1)
    cand_logits = logits.gather(1, cand_ids)
    probs = F.softmax(cand_logits, dim=-1)
    gold_logit = cand_logits[:, 0]
    max_comp = cand_logits[:, 1:].max(dim=1).values
    margins = gold_logit - max_comp
    labels_local = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)
    return probs, margins, labels_local


def r3f_kl_logits(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    detach_target: bool = True,
    labels: Optional[torch.Tensor] = None,
    spectral_cfg: Optional[Dict] = None,
) -> torch.Tensor:
    if detach_target:
        clean_logits = clean_logits.detach()
    clean_logp = F.log_softmax(clean_logits, dim=-1)
    noisy_logp = F.log_softmax(noisy_logits, dim=-1)
    clean_p = clean_logp.exp()
    kl = (clean_p * (clean_logp - noisy_logp)).sum(dim=-1)
    loss = kl

    if labels is not None and spectral_cfg is not None and spectral_cfg.get("inner_guided", False):
        alpha = float(spectral_cfg.get("alpha", 0.1))
        gamma = float(spectral_cfg.get("gamma", 0.2))
        tau = float(spectral_cfg.get("tau", 0.1))
        top_k = int(spectral_cfg.get("top_k", 10))

        if noisy_logits.dim() == 2:
            noisy_probs = F.softmax(noisy_logits, dim=-1)
            with torch.no_grad():
                gold_probs = noisy_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                other_probs = noisy_probs.clone()
                other_probs.scatter_(1, labels.unsqueeze(1), -1.0)
                max_other = other_probs.max(dim=1).values
                margins = gold_probs - max_other
            batch_mat = batch_transition_matrix(noisy_probs, labels, margins, gamma, tau)
            spectral_risk = batch_mat.sum(dim=0).max()
            loss = loss + alpha * spectral_risk
        elif noisy_logits.dim() == 3:
            valid_mask = labels != -100
            if valid_mask.any():
                logits_v = noisy_logits[valid_mask]
                gold_v = labels[valid_mask].long()
                probs, margins, labels_local = _candidate_probs_from_logits(logits_v, gold_v, top_k=top_k)
                batch_mat = batch_transition_matrix(probs, labels_local, margins, gamma, tau)
                spectral_risk = batch_mat.sum(dim=0).max()
                loss = loss + alpha * spectral_risk
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()
