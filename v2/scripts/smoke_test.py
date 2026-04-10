#!/usr/bin/env python3
"""
Quick sanity check without any GPU training.
Tests: config, data loading, perturbation, plugin math, eval logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config  import get_cfg
from data    import load_arc
from perturb import apply_perturbation, random_perturbation
from plugin  import (compute_margins, compute_gates, build_transition_matrix,
                     r_spec, r_stab, plugin_loss,
                     compute_vwr_gamma, compute_sigma_max, compute_fragile_ratio)


def test_config():
    cfg = get_cfg()
    assert cfg["num_classes"] == 4
    assert len(cfg["label_chars"]) == 4
    print("  [OK] config")


def test_data():
    cfg  = get_cfg(train_size=50, val_size=20, test_size=20)
    data = load_arc(cfg, seed=42)
    assert len(data["train"]) <= 50 and len(data["train"]) > 0
    assert len(data["val"])   <= 20 and len(data["val"])   > 0
    assert len(data["test"])  <= 20 and len(data["test"])  > 0
    ex = data["train"][0]
    assert "text"   in ex
    assert "prompt" in ex
    assert "label"  in ex
    assert 0 <= ex["label"] < 4
    print("  [OK] data loading")


def test_perturb():
    cfg  = get_cfg()
    data = load_arc(cfg, seed=42)
    ex   = data["train"][0]

    for ptype in ["typo", "distractor", "format_rewrite"]:
        ex2 = apply_perturbation(ex, ptype, cfg)
        assert "prompt" in ex2
        assert isinstance(ex2["prompt"], str)
    print("  [OK] perturbations")


def test_plugin():
    torch.manual_seed(0)
    B, K = 8, 4
    logits_c = torch.randn(B, K)
    logits_p = torch.randn(B, K, requires_grad=True)
    labels   = torch.randint(0, K, (B,))
    gamma, kappa = 0.3, 0.5

    margins = compute_margins(logits_p, labels)
    assert margins.shape == (B,)

    gates = compute_gates(margins, gamma, kappa)
    assert gates.shape == (B,)
    assert (gates >= 0).all() and (gates <= 1).all()

    T = build_transition_matrix(logits_p, labels, gates, K)
    assert T.shape == (K, K)
    diag = T.diagonal()
    assert (diag == 0).all(), "Diagonal must be zero"

    rs  = r_spec(T)
    rst = r_stab(logits_c, logits_p)
    assert rs  >= 0
    assert rst >= 0

    rs_v, rst_v, reg = plugin_loss(logits_c, logits_p, labels,
                                   gamma, kappa, K, 0.1, 0.05)
    reg.backward()
    assert logits_p.grad is not None

    # Eval metrics
    with torch.no_grad():
        vwr  = compute_vwr_gamma(logits_p, labels, gamma, kappa, K)
        smax = compute_sigma_max(logits_p, labels, gamma, kappa, K)
        frag = compute_fragile_ratio(logits_p, labels, gamma, kappa)
        assert vwr  >= 0
        assert smax >= 0
        assert 0 <= frag <= 1

    print("  [OK] plugin math + gradients")


def test_trainers_cpu():
    """Instantiate all trainers, ensure no import error."""
    from trainers import CleanStep, AugStep, R3FStep, SMARTStep, AWPStep
    from trainers import PluginWrapper, make_trainer
    cfg = get_cfg()
    s = make_trainer("base_clean")
    s = make_trainer("base_aug")
    s = make_trainer("r3f")
    s = make_trainer("smart")
    s = make_trainer("awp")
    s = make_trainer("plugin",       gamma=0.3, cfg=cfg)
    s = make_trainer("r3f_plugin",   gamma=0.3, cfg=cfg)
    s = make_trainer("smart_plugin", gamma=0.3, cfg=cfg)
    s = make_trainer("awp_plugin",   gamma=0.3, cfg=cfg)
    print("  [OK] trainer instantiation")


if __name__ == "__main__":
    print("Running smoke tests …")
    test_config()
    test_data()
    test_perturb()
    test_plugin()
    test_trainers_cpu()
    print("\n  All smoke tests PASSED.")
