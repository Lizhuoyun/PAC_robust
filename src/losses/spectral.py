from typing import Optional

import torch
import torch.nn.functional as F


def batch_transition_matrix(
    probs: torch.Tensor,
    labels: torch.Tensor,
    margins: torch.Tensor,
    gamma: float,
    tau: float,
) -> torch.Tensor:
    """Eq.(30): margin-aware transition estimator; diagonal zero."""
    num_classes = probs.size(1)
    # When the model runs in bf16/fp16, `probs` can be low-precision.
    # Compute the transition matrix in fp32 for stability and to avoid
    # dtype mismatch with the one-hot tensor.
    gate = torch.sigmoid((gamma - margins) / tau).float()
    probs_f = probs.float()
    one_hot = F.one_hot(labels, num_classes=num_classes).to(device=probs.device, dtype=torch.float32)
    counts = one_hot.sum(dim=0).clamp_min(1.0)
    weighted = probs_f * gate.unsqueeze(1)
    mat = weighted.t() @ one_hot
    mat = mat / counts.unsqueeze(0)
    mat = mat * (1.0 - torch.eye(num_classes, device=probs.device, dtype=mat.dtype))
    return mat


class SpectralEMA:
    def __init__(self, num_classes: int, beta_ema: float, device: torch.device, n_refresh: int = 1):
        self.beta_ema = beta_ema
        self.mat = torch.zeros(num_classes, num_classes, device=device)
        self.eigvec = torch.ones(num_classes, device=device)
        self.n_refresh = max(int(n_refresh), 1)
        self.step = 0

    def update(self, batch_mat: torch.Tensor) -> None:
        # Detach the running state to avoid backprop through previous steps' graphs.
        # Keep gradient only for the current batch contribution.
        self.mat = self.beta_ema * self.mat.detach() + (1.0 - self.beta_ema) * batch_mat
        self.step += 1

    def sigma_max(self, t_pi: int) -> torch.Tensor:
        """Eq.(27): sigma_max via power iteration on A^T A; stop-grad through eigvec."""
        A = self.mat
        v = self.eigvec.detach()
        if self.step % self.n_refresh == 0:
            for _ in range(t_pi):
                Av = A @ v
                AtAv = A.t() @ Av
                v = AtAv / AtAv.norm(p=2).clamp_min(1e-6)
            self.eigvec = v.detach()
        sigma = (A @ v).norm(p=2)
        return sigma


def stability_kl(
    logits_clean: torch.Tensor,
    logits_noisy: torch.Tensor,
) -> torch.Tensor:
    """Eq.(35): stability regularizer KL(p_phi || p_{phi+u})."""
    logp = F.log_softmax(logits_clean, dim=-1)
    logq = F.log_softmax(logits_noisy, dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1).mean()


@torch.no_grad()
def sample_lora_noise(params, sigma: float):
    return [torch.randn_like(p) * sigma for p in params]


def apply_lora_noise(params, noise):
    for p, n in zip(params, noise):
        p.data.add_(n)


def remove_lora_noise(params, noise):
    for p, n in zip(params, noise):
        p.data.sub_(n)
