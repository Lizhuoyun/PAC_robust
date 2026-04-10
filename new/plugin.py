"""
Gamma-aware plugin regulariser.

Core idea:
  1. Compute true-label margin on perturbed logits.
  2. Gate each sample by sigmoid((gamma - margin) / kappa).
  3. Build a margin-aware transition matrix T  (rows = wrong class, cols = true class).
  4. R_spec  = sigma_max(T)   — spectral penalty.
  5. R_stab  = sym-KL(clean || perturbed) — stability penalty.
  6. total   = base_loss + alpha * R_spec + beta * R_stab
"""
import torch
import torch.nn.functional as F


def compute_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """True-label margin: p(y_true) - max_{k!=y} p(k)."""
    probs = F.softmax(logits, dim=-1)
    batch_idx = torch.arange(len(labels), device=logits.device)
    true_probs = probs[batch_idx, labels]

    mask = torch.ones_like(probs, dtype=torch.bool)
    mask[batch_idx, labels] = False
    max_other = probs.masked_fill(~mask, -1e9).max(dim=-1).values

    return true_probs - max_other


def compute_gates(margins: torch.Tensor, gamma: float, kappa: float) -> torch.Tensor:
    return torch.sigmoid((gamma - margins) / kappa)


def build_transition_matrix(logits: torch.Tensor, labels: torch.Tensor,
                            gates: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    T[k, j] = sum_{i: y_i=j} gate_i * p(k | x'_i)    with T[j,j] = 0.
    """
    probs = F.softmax(logits, dim=-1)
    weighted = probs * gates.unsqueeze(1)  # (B, K)

    T = torch.zeros(num_classes, num_classes, device=logits.device, dtype=logits.dtype)
    for j in range(num_classes):
        mask = labels == j
        if mask.any():
            T[:, j] = weighted[mask].sum(0)
    T.fill_diagonal_(0)
    return T


def r_spec(T: torch.Tensor) -> torch.Tensor:
    """Largest singular value of the transition matrix."""
    sv = torch.linalg.svdvals(T)
    return sv[0] if sv.numel() > 0 else torch.tensor(0.0, device=T.device)


def r_stab(clean_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between clean and perturbed probability distributions."""
    p = F.log_softmax(clean_logits, dim=-1)
    q = F.log_softmax(perturbed_logits, dim=-1)
    p_prob = F.softmax(clean_logits, dim=-1)
    q_prob = F.softmax(perturbed_logits, dim=-1)
    kl_pq = F.kl_div(q, p_prob, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p, q_prob, reduction="batchmean", log_target=False)
    return (kl_pq + kl_qp) / 2


def plugin_loss(clean_logits: torch.Tensor, perturbed_logits: torch.Tensor,
                labels: torch.Tensor, gamma: float, kappa: float,
                num_classes: int, alpha: float, beta: float):
    """
    Returns (R_spec_val, R_stab_val, total_reg) where
    total_reg = alpha * R_spec + beta * R_stab.
    """
    margins = compute_margins(perturbed_logits, labels)
    gates = compute_gates(margins, gamma, kappa)
    T = build_transition_matrix(perturbed_logits, labels, gates, num_classes)
    rs = r_spec(T)
    rst = r_stab(clean_logits.detach(), perturbed_logits) if beta > 0 else torch.tensor(0.0, device=perturbed_logits.device)
    reg = alpha * rs + beta * rst
    return rs, rst, reg


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_vwr_gamma(logits: torch.Tensor, labels: torch.Tensor,
                      gamma: float, kappa: float, num_classes: int) -> float:
    """VWR_gamma = max column-sum (1-norm) of the transition matrix."""
    margins = compute_margins(logits, labels)
    gates = compute_gates(margins, gamma, kappa)
    T = build_transition_matrix(logits, labels, gates, num_classes)
    return T.abs().sum(0).max().item()


def compute_sigma_max(logits: torch.Tensor, labels: torch.Tensor,
                      gamma: float, kappa: float, num_classes: int) -> float:
    margins = compute_margins(logits, labels)
    gates = compute_gates(margins, gamma, kappa)
    T = build_transition_matrix(logits, labels, gates, num_classes)
    return r_spec(T).item()
