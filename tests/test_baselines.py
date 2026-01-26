import torch

from src.losses.r3f import r3f_kl_logits
from src.losses.smart import smart_kl
from src.train.utils import resolve_presets


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10, hidden=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden)
        self.proj = torch.nn.Linear(hidden, vocab_size)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)
        logits = self.proj(inputs_embeds)
        return type("Out", (), {"logits": logits})


def test_r3f_zero_noise():
    logits = torch.randn(4, 5)
    kl = r3f_kl_logits(logits, logits, mask=None, detach_target=True)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_smart_kl_increases_with_eps():
    torch.manual_seed(0)
    model = DummyModel(vocab_size=12, hidden=6)
    input_ids = torch.randint(0, 12, (2, 4))
    attention_mask = torch.ones_like(input_ids)

    def logits_fn(ids, mask, inputs_embeds=None):
        return model(input_ids=ids, attention_mask=mask, inputs_embeds=inputs_embeds).logits

    kl_small = smart_kl(
        model,
        input_ids,
        attention_mask,
        logits_fn=logits_fn,
        steps=1,
        step_size=0.01,
        epsilon=0.01,
        norm="l2",
        detach_target=True,
    )
    kl_big = smart_kl(
        model,
        input_ids,
        attention_mask,
        logits_fn=logits_fn,
        steps=2,
        step_size=0.05,
        epsilon=0.1,
        norm="l2",
        detach_target=True,
    )
    assert kl_big >= kl_small


def test_resolve_presets():
    cfg = {
        "r3f": {
            "preset": "cheap",
            "presets": {"cheap": {"noise_std": 0.01, "lambda": 0.5}},
            "noise_std": 0.02,
            "lambda": 1.0,
        }
    }
    out = resolve_presets(cfg)
    assert out["r3f"]["noise_std"] == 0.01
    assert out["r3f"]["lambda"] == 0.5
