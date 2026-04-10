"""
All trainer step implementations.

Each trainer exposes a single method:
    step(model, tok, batch, cfg, rng, label_ids, device)
        → (loss, metrics, clean_logits, pert_logits, labels)

  clean_logits : (B, K) detached   — only used for plugin eval / R_stab reference
  pert_logits  : (B, K) IN GRAPH   — plugin gradient flows through this tensor
  loss         : scalar tensor with grad_fn

Design principle
----------------
The plugin regulariser must see the SAME perturbed distribution that each base
method constructs.  To achieve this, every inner step returns pert_logits
**in the computational graph** (no .detach()).  PluginWrapper then computes

    total_loss = inner_loss + alpha * R_spec(pert_logits)
                            + beta  * R_stab(pert_logits)

in a single backward pass.  No separate perturbation is ever re-generated.

Trainers:
  CleanStep     — CE(clean),        pert=clean (detached, plugin won't regularise)
  AugStep       — (CE_clean+CE_pert)/2,    pert_logits = text-perturbed logits (in graph)
  R3FStep       — CE + λ·KL(p_clean‖p_noise), pert_logits = embedding-noisy logits (in graph)
  SMARTStep     — CE + λ·KL(p_clean‖p_adv),   pert_logits = adv-emb logits (in graph)
  AWPStep       — (CE_clean+CE_adv)/2,     pert_logits = weight-perturbed logits (in graph)

  PluginWrapper — wraps any inner step, adds plugin loss directly on pert_logits.
"""
import torch
import torch.nn.functional as F

from models import (
    tokenise_batch, extract_class_logits,
    get_embed_layer, forward_from_embeds,
)
from perturb import random_perturbation
from plugin import plugin_loss


# ── Utility ────────────────────────────────────────────────────────────────

def _encode(tok, examples, cfg, device):
    prompts = [e["prompt"]      for e in examples]
    lcs     = [e["label_char"]  for e in examples]
    enc     = tokenise_batch(tok, prompts, lcs, cfg["max_seq_len"])
    return {k: v.to(device) for k, v in enc.items()}


def _forward(model, enc, label_ids):
    out = model(input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"])
    return extract_class_logits(out.logits, enc["label_positions"], label_ids)


def _labels(examples, device):
    return torch.tensor([e["label"] for e in examples], device=device)


# ── CleanStep ──────────────────────────────────────────────────────────────

class CleanStep:
    """Standard cross-entropy on clean data only (Base-clean)."""

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        enc    = _encode(tok, batch, cfg, device)
        labs   = _labels(batch, device)
        logits = _forward(model, enc, label_ids)
        loss   = F.cross_entropy(logits, labs)
        # No perturbation: pert_logits = clean (detached).
        # PluginWrapper will add no meaningful gradient here.
        return (loss,
                {"ce": loss.item()},
                logits.detach(),
                logits.detach(),
                labs)


# ── AugStep ────────────────────────────────────────────────────────────────

class AugStep:
    """
    Data-augmentation baseline (Base-aug).
    Loss = (CE_clean + CE_text_pert) / 2

    pert_logits: text-perturbed logits — IN the computation graph.
    PluginWrapper will add R_spec/R_stab on this same pert distribution.
    """

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        # Clean forward
        enc_c  = _encode(tok, batch, cfg, device)
        labs   = _labels(batch, device)
        cl_log = _forward(model, enc_c, label_ids)
        ce_c   = F.cross_entropy(cl_log, labs)

        # Text-perturbed forward (in graph — no detach)
        pert_batch = [random_perturbation(ex, cfg, rng) for ex in batch]
        enc_p      = _encode(tok, pert_batch, cfg, device)
        pt_log     = _forward(model, enc_p, label_ids)   # <-- in graph
        ce_p       = F.cross_entropy(pt_log, labs)

        loss = (ce_c + ce_p) / 2.0
        return (loss,
                {"ce": loss.item(), "ce_clean": ce_c.item(), "ce_pert": ce_p.item()},
                cl_log.detach(),
                pt_log,           # in graph: plugin gradient flows here
                labs)


# ── R3FStep ────────────────────────────────────────────────────────────────

class R3FStep:
    """
    R3F (Robust Representations through Regularized Fine-Tuning).
    L = CE(clean) + λ · KL( p_clean ‖ p_emb_noise )

    The embedding noise is sampled once.  The resulting noisy_log is used for
    both the KL term (in inner loss) and the plugin terms — same distribution,
    single forward pass.

    pert_logits: noisy-embedding logits — IN the computation graph.
    """

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        enc  = _encode(tok, batch, cfg, device)
        labs = _labels(batch, device)

        # Clean forward
        cl_log = _forward(model, enc, label_ids)
        ce     = F.cross_entropy(cl_log, labs)

        # Embedding-space noise — computed once, used in both KL and plugin
        embed = get_embed_layer(model)
        with torch.no_grad():
            embeds = embed(enc["input_ids"])          # (B, T, H)
        noise     = torch.randn_like(embeds) * cfg["r3f_noise_std"]
        noisy_log = forward_from_embeds(
            model, embeds + noise,
            enc["attention_mask"], enc["label_positions"], label_ids)
        # noisy_log is in the computation graph w.r.t. model parameters

        p_clean = F.softmax(cl_log.detach(), dim=-1)
        kl = F.kl_div(F.log_softmax(noisy_log, dim=-1), p_clean,
                      reduction="batchmean", log_target=False)

        loss = ce + cfg["r3f_lambda"] * kl
        return (loss,
                {"ce": ce.item(), "r3f_kl": kl.item()},
                cl_log.detach(),
                noisy_log,        # in graph: plugin regularises the SAME noise
                labs)


# ── SMARTStep ──────────────────────────────────────────────────────────────

class SMARTStep:
    """
    SMART (Smoothness-Inducing Adversarial Regularization and Training).
    L = CE(clean) + λ · KL( p_clean ‖ p_adv_emb )

    PGD inner loop uses torch.autograd.grad(kl, delta) to compute gradient
    w.r.t. the perturbation ONLY — model parameters do NOT accumulate spurious
    gradients during the search phase.

    After PGD, a single clean forward pass with the found delta gives adv_log
    in the computation graph.  This same adv_log is used for both the KL term
    and the plugin terms.

    pert_logits: adversarial-embedding logits — IN the computation graph.
    """

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        enc  = _encode(tok, batch, cfg, device)
        labs = _labels(batch, device)

        # Clean forward
        cl_log = _forward(model, enc, label_ids)
        ce     = F.cross_entropy(cl_log, labs)
        p_ref  = F.softmax(cl_log.detach(), dim=-1)

        embed = get_embed_layer(model)
        with torch.no_grad():
            base_embeds = embed(enc["input_ids"])     # (B, T, H)

        eps       = cfg["smart_epsilon"]
        step_size = cfg["smart_step_size"]
        n_steps   = cfg["smart_steps"]

        # PGD: maximise KL( p_ref || p_{emb+δ} ) over δ.
        # torch.autograd.grad avoids accumulating model gradients.
        delta = torch.zeros_like(base_embeds).normal_(0, eps * 0.1)
        delta.requires_grad_(True)

        for _ in range(n_steps):
            adv_log_search = forward_from_embeds(
                model, base_embeds + delta,
                enc["attention_mask"], enc["label_positions"], label_ids)
            kl_search = F.kl_div(
                F.log_softmax(adv_log_search, dim=-1), p_ref,
                reduction="batchmean", log_target=False)
            # Use .backward() (compatible with gradient checkpointing).
            # Accumulates grads in model params too — zeroed right after.
            kl_search.backward()
            with torch.no_grad():
                d = delta + step_size * delta.grad.sign()
                d = torch.clamp(d, -eps, eps)
            # Zero out spurious model gradients from PGD search
            model.zero_grad(set_to_none=True)
            delta = d.detach().requires_grad_(True)

        # Final forward with the found delta — IN the computation graph
        adv_log = forward_from_embeds(
            model, base_embeds + delta.detach(),
            enc["attention_mask"], enc["label_positions"], label_ids)

        kl = F.kl_div(
            F.log_softmax(adv_log, dim=-1), p_ref,
            reduction="batchmean", log_target=False)

        loss = ce + cfg["smart_lambda"] * kl
        return (loss,
                {"ce": ce.item(), "smart_kl": kl.item()},
                cl_log.detach(),
                adv_log,          # in graph: plugin regularises the SAME adv emb
                labs)


# ── AWPStep ────────────────────────────────────────────────────────────────

class AWPStep:
    """
    AWP (Adversarial Weight Perturbation) — LoRA-subspace approximation.

    1. Compute gradient of CE_clean w.r.t. LoRA parameters (via autograd.grad,
       so model.grad buffers are NOT polluted at this stage).
    2. Perturb LoRA weights in-place: w_adv = clamp(w + lr*g/‖g‖, w±ε).
    3. Forward pass under perturbed weights — IN the computation graph.
    4. Restore original weights.
    5. Loss = (CE_clean + CE_adv) / 2.

    Because p.data assignments are not tracked by autograd, backward() at step 5
    evaluates gradients at the RESTORED weight values.  For small perturbations
    this is the standard AWP approximation (acceptable in practice).

    pert_logits: weight-perturbed logits — IN the computation graph.
    """

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        enc  = _encode(tok, batch, cfg, device)
        labs = _labels(batch, device)

        # ── Step 1: clean forward ────────────────────────────────────────
        cl_log = _forward(model, enc, label_ids)
        ce_c   = F.cross_entropy(cl_log, labs)

        # ── Step 2: gradient w.r.t. LoRA params (no model.grad pollution) ─
        lora_params = [(n, p) for n, p in model.named_parameters()
                       if "lora_" in n and p.requires_grad]
        if not lora_params:
            # Fallback: no LoRA params — behave like AugStep
            pert_batch = [random_perturbation(ex, cfg, rng) for ex in batch]
            enc_p  = _encode(tok, pert_batch, cfg, device)
            pt_log = _forward(model, enc_p, label_ids)
            ce_p   = F.cross_entropy(pt_log, labs)
            loss   = (ce_c + ce_p) / 2.0
            return (loss,
                    {"ce": loss.item()},
                    cl_log.detach(), pt_log, labs)

        grads = torch.autograd.grad(
            ce_c, [p for _, p in lora_params],
            retain_graph=True, allow_unused=True)

        # ── Step 3: perturb LoRA weights ─────────────────────────────────
        adv_lr  = cfg["awp_adv_lr"]
        adv_eps = cfg["awp_adv_eps"]
        backup  = {}
        for (n, p), g in zip(lora_params, grads):
            if g is None:
                continue
            backup[n] = p.data.clone()
            norm      = g.norm() + 1e-8
            p.data    = p.data + adv_lr * g / norm
            p.data    = torch.clamp(p.data,
                                    backup[n] - adv_eps,
                                    backup[n] + adv_eps)

        # ── Step 4: forward under perturbed weights — IN graph ────────────
        # No torch.no_grad() here: adv_log must carry gradients so that
        # plugin_loss can backprop through R_spec(adv_log).
        adv_log = _forward(model, enc, label_ids)
        ce_adv  = F.cross_entropy(adv_log, labs)

        # ── Step 5: restore weights ───────────────────────────────────────
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup:
                    p.data = backup[n]

        loss = (ce_c + ce_adv) / 2.0
        return (loss,
                {"ce": loss.item(), "ce_clean": ce_c.item(), "ce_adv": ce_adv.item()},
                cl_log.detach(),
                adv_log,          # in graph: plugin regularises weight-perturbed dist
                labs)


# ── PluginWrapper ──────────────────────────────────────────────────────────

class PluginWrapper:
    """
    Composable plugin regulariser.

    Calls the inner step, then adds plugin loss directly on the inner step's
    pert_logits — which is already in the computation graph.  No separate
    perturbation is ever re-generated.

        total = inner_loss
              + alpha * R_spec(pert_logits, gates, T)
              + beta  * R_stab(clean_logits, pert_logits, gates)

    The gradient of R_spec and R_stab flows through pert_logits back into the
    model weights via the same computation graph as the inner_loss.  Plugin
    strength is therefore directly controlled by alpha and beta.
    """

    def __init__(self, inner, gamma: float, alpha: float, beta: float,
                 kappa: float, K: int):
        self.inner = inner
        self.gamma = gamma
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa
        self.K     = K

    def __call__(self, model, tok, batch, cfg, rng, label_ids, device):
        loss, metrics, cl_log, pt_log, labs = self.inner(
            model, tok, batch, cfg, rng, label_ids, device)

        # pt_log is already in the computation graph (returned by inner step).
        # cl_log is detached (only used as reference distribution).
        rs, rst, reg = plugin_loss(
            cl_log,       # detached reference
            pt_log,       # in graph — gradient of R_spec/R_stab flows here
            labs,
            gamma=self.gamma,
            kappa=self.kappa,
            K=self.K,
            alpha=self.alpha,
            beta=self.beta,
        )

        total = loss + reg
        m2    = dict(metrics)
        m2.update(r_spec=rs.item(), r_stab=rst.item(), reg=reg.item())
        return (total, m2, cl_log, pt_log.detach(), labs)


# ── Factory ────────────────────────────────────────────────────────────────

_INNER_MAP = {
    "base_clean"  : CleanStep,
    "base_aug"    : AugStep,
    "r3f"         : R3FStep,
    "smart"       : SMARTStep,
    "awp"         : AWPStep,
}

def make_trainer(method: str, gamma: float = None, cfg: dict = None):
    """
    Create the appropriate trainer for a given method name.

    method examples: 'base_clean', 'base_aug', 'plugin',
                     'r3f', 'r3f_plugin', 'smart', 'smart_plugin',
                     'awp', 'awp_plugin'
    """
    if method.endswith("_plugin"):
        inner_name = method[: -len("_plugin")]
        inner      = _INNER_MAP[inner_name]()
        assert gamma is not None, "gamma required for plugin"
        return PluginWrapper(
            inner=inner, gamma=gamma,
            alpha=cfg["alpha"], beta=cfg["beta"],
            kappa=cfg["kappa"], K=cfg["num_classes"],
        )
    elif method == "plugin":
        inner = AugStep()
        assert gamma is not None
        return PluginWrapper(
            inner=inner, gamma=gamma,
            alpha=cfg["alpha"], beta=cfg["beta"],
            kappa=cfg["kappa"], K=cfg["num_classes"],
        )
    else:
        return _INNER_MAP[method]()
