"""
Gamma-aware plugin regulariser core.

L_plugin = alpha * R_spec + beta * R_stab

where:
  R_spec  = sigma_max( T )   — spectral norm of margin-aware transition matrix
  R_stab  = sym-KL(clean || perturbed) — output stability penalty

This module is model-agnostic: it only operates on (B, K) logit tensors.
"""
import torch
import torch.nn.functional as F


# ── Core computations ──────────────────────────────────────────────────────

def compute_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    True-label margin: p(y) - max_{k ≠ y} p(k).
    logits : (B, K)
    labels : (B,)
    returns: (B,)

    Fully differentiable — uses additive masking instead of in-place assignment
    so that gradients flow correctly when logits is in the computation graph.
    """
    probs   = F.softmax(logits, dim=-1)             # (B, K)
    B, K    = probs.shape
    idx     = torch.arange(B, device=logits.device)
    true_p  = probs[idx, labels]                    # (B,)

    # Subtract a large constant from the true-class column (no in-place op)
    eye     = F.one_hot(labels, K).to(dtype=probs.dtype, device=probs.device)
    other_p = (probs - eye * 1e9).max(dim=-1).values
    return true_p - other_p


def compute_gates(margins: torch.Tensor, gamma: float, kappa: float) -> torch.Tensor:
    """
    Gamma-aware gate: sigmoid((gamma - margin) / kappa).
    High gate value  ⟹ near-boundary (fragile) sample.
    """
    return torch.sigmoid((gamma - margins) / kappa)


def build_transition_matrix(logits: torch.Tensor, labels: torch.Tensor,
                             gates: torch.Tensor, K: int) -> torch.Tensor:
    """
    T[k, j] = Σ_{i: y_i=j} gate_i · p(k | x'_i),  T[j,j] = 0.
    Shape: (K, K)

    Fully differentiable via matrix multiplication:
        T = (probs * gates).T  @  one_hot(labels)
    No Python loops, no in-place assignments — gradient flows correctly
    from r_spec(T) back through gates and probs to logits.
    """
    probs    = F.softmax(logits, dim=-1)                          # (B, K)
    weighted = probs * gates.unsqueeze(1)                         # (B, K)
    label_oh = F.one_hot(labels, K).to(dtype=probs.dtype,
                                        device=probs.device)      # (B, K)
    T = weighted.t() @ label_oh                                   # (K, K)
    # Zero diagonal: T[j,j] is self-to-self "corruption" which is excluded
    T = T * (1.0 - torch.eye(K, device=T.device, dtype=T.dtype))
    return T


def r_spec(T: torch.Tensor) -> torch.Tensor:
    """σ_max(T) — largest singular value (spectral norm)."""
    sv = torch.linalg.svdvals(T)
    return sv[0] if sv.numel() > 0 else T.new_tensor(0.0)


def r_stab(logits_clean: torch.Tensor,
           logits_pert: torch.Tensor) -> torch.Tensor:
    """
    Symmetric KL divergence: (KL(p||q) + KL(q||p)) / 2.
    """
    lp = F.log_softmax(logits_clean, dim=-1)
    lq = F.log_softmax(logits_pert,  dim=-1)
    p  = lp.exp()
    q  = lq.exp()
    kl_pq = F.kl_div(lq, p, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(lp, q, reduction="batchmean", log_target=False)
    return (kl_pq + kl_qp) / 2.0


# ── Plugin loss ────────────────────────────────────────────────────────────

def plugin_loss(logits_clean: torch.Tensor,
                logits_pert: torch.Tensor,
                labels: torch.Tensor,
                gamma: float,
                kappa: float,
                K: int,
                alpha: float,
                beta: float):
    """
    Compute plugin regularisation and return (R_spec_val, R_stab_val, total_reg).

    logits_clean : (B, K) — clean-input logits (detached or not)
    logits_pert  : (B, K) — perturbed-input logits (in computational graph)
    """
    margins = compute_margins(logits_pert, labels)
    gates   = compute_gates(margins, gamma, kappa)

    T   = build_transition_matrix(logits_pert, labels, gates, K)
    rs  = r_spec(T)

    if beta > 0.0:
        rst = r_stab(logits_clean.detach(), logits_pert)
    else:
        rst = logits_pert.new_tensor(0.0)

    reg = alpha * rs + beta * rst
    return rs, rst, reg


# ── Evaluation helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_vwr_gamma(logits: torch.Tensor, labels: torch.Tensor,
                      gamma: float, kappa: float, K: int) -> float:
    """
    VWR_gamma = max column-sum (1-norm) of T.
    """
    margins = compute_margins(logits, labels)
    gates   = compute_gates(margins, gamma, kappa)
    T       = build_transition_matrix(logits, labels, gates, K)
    return T.abs().sum(0).max().item()


@torch.no_grad()
def compute_sigma_max(logits: torch.Tensor, labels: torch.Tensor,
                      gamma: float, kappa: float, K: int) -> float:
    margins = compute_margins(logits, labels)
    gates   = compute_gates(margins, gamma, kappa)
    T       = build_transition_matrix(logits, labels, gates, K)
    return r_spec(T).item()


@torch.no_grad()
def compute_fragile_ratio(logits: torch.Tensor, labels: torch.Tensor,
                          gamma: float, kappa: float,
                          threshold: float = 0.5) -> float:
    """Fraction of samples with gate > threshold (near-boundary fragile)."""
    margins = compute_margins(logits, labels)
    gates   = compute_gates(margins, gamma, kappa)
    return (gates > threshold).float().mean().item()


@torch.no_grad()
def compute_mean_gate(logits: torch.Tensor, labels: torch.Tensor,
                      gamma: float, kappa: float) -> float:
    margins = compute_margins(logits, labels)
    gates   = compute_gates(margins, gamma, kappa)
    return gates.mean().item()
