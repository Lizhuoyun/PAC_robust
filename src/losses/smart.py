from typing import Callable, Optional, Dict, Tuple

import torch
import torch.nn.functional as F

from src.losses.spectral import batch_transition_matrix


def _candidate_probs_from_logits(
    logits: torch.Tensor, gold_ids: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a small candidate distribution per position: [gold] + top_k competitors.
    Returns:
      probs: (N, top_k+1)
      margins: (N,)  (gold_logit - max_competitor_logit)
      labels_local: (N,) all zeros (gold at index 0)
    """
    # logits: (N, V), gold_ids: (N,)
    masked = logits.clone()
    masked[torch.arange(logits.size(0), device=logits.device), gold_ids] = -1e9
    topk = torch.topk(masked, k=top_k, dim=-1)
    cand_ids = torch.cat([gold_ids.unsqueeze(1), topk.indices], dim=1)  # (N, top_k+1)
    cand_logits = logits.gather(1, cand_ids)  # (N, top_k+1)
    probs = F.softmax(cand_logits, dim=-1)
    # margin in logits space between gold and strongest competitor
    gold_logit = cand_logits[:, 0]
    max_comp = cand_logits[:, 1:].max(dim=1).values
    margins = gold_logit - max_comp
    labels_local = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)
    return probs, margins, labels_local


def _kl_from_logits(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    clean_logp = F.log_softmax(clean_logits, dim=-1)
    adv_logp = F.log_softmax(adv_logits, dim=-1)
    clean_p = clean_logp.exp()
    kl = (clean_p * (clean_logp - adv_logp)).sum(dim=-1)
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp_min(1.0)
    return kl.mean()


def smart_kl(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_fn: Callable[..., torch.Tensor],
    steps: int,
    step_size: float,
    epsilon: float,
    norm: str = "l2",
    mask: Optional[torch.Tensor] = None,
    detach_target: bool = True,
    # New arguments for spectral guidance
    labels: Optional[torch.Tensor] = None,
    spectral_cfg: Optional[Dict] = None,
) -> torch.Tensor:
    with torch.no_grad():
        clean_logits = logits_fn(input_ids, attention_mask)
    if detach_target:
        clean_logits = clean_logits.detach()

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)
    delta = torch.zeros_like(inputs_embeds, requires_grad=True)

    for _ in range(steps):
        adv_logits = logits_fn(input_ids, attention_mask, inputs_embeds=inputs_embeds + delta)
        
        # Primary objective: KL divergence
        loss = _kl_from_logits(clean_logits, adv_logits, mask=mask)
        
        # Optional: Spectral guidance in the inner loop
        if labels is not None and spectral_cfg is not None and spectral_cfg.get("inner_guided", False):
            alpha = spectral_cfg.get("alpha", 0.1)
            gamma = spectral_cfg.get("gamma", 0.2)
            tau = spectral_cfg.get("tau", 0.1)
            top_k = int(spectral_cfg.get("top_k", 10))
            
            # Classification: logits (B, C), labels (B,)
            # Generation: logits (B, T, V), labels (B, T) with -100 mask
            if adv_logits.dim() == 2:
                adv_probs = F.softmax(adv_logits, dim=-1)
                # margins in prob space (ok for classification guidance)
                with torch.no_grad():
                    gold_probs = adv_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                    other_probs = adv_probs.clone()
                    other_probs.scatter_(1, labels.unsqueeze(1), -1.0)
                    max_other = other_probs.max(dim=1).values
                    margins = gold_probs - max_other
                batch_mat = batch_transition_matrix(adv_probs, labels, margins, gamma, tau)
                spectral_risk = batch_mat.sum(dim=0).max()
                loss = loss + alpha * spectral_risk
            elif adv_logits.dim() == 3:
                # Use top-k candidate distribution to avoid huge vocab matrix
                valid_mask = labels != -100
                if valid_mask.any():
                    logits_v = adv_logits[valid_mask]  # (N, V)
                    gold_v = labels[valid_mask].long()  # (N,)
                    probs, margins, labels_local = _candidate_probs_from_logits(logits_v, gold_v, top_k=top_k)
                    batch_mat = batch_transition_matrix(probs, labels_local, margins, gamma, tau)
                    spectral_risk = batch_mat.sum(dim=0).max()
                    loss = loss + alpha * spectral_risk

        grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
        if norm == "l2":
            grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            delta.data = delta.data + step_size * grad / grad_norm
            delta_norm = delta.data.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            factor = torch.clamp(delta_norm / epsilon, min=1.0)
            delta.data = delta.data / factor
        else:
            delta.data = delta.data + step_size * grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

    adv_logits = logits_fn(input_ids, attention_mask, inputs_embeds=inputs_embeds + delta.detach())
    return _kl_from_logits(clean_logits, adv_logits, mask=mask)
