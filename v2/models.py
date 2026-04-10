"""
Model loading: Qwen2.5-0.5B-Instruct (pure text causal LM), 4-bit + LoRA.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, get_peft_model, PeftModel,
    TaskType, prepare_model_for_kbit_training,
)


# ── Base model loading ─────────────────────────────────────────────────────

def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_base(cfg: dict):
    """Load quantised base model + tokenizer (no LoRA)."""
    model_name = cfg["model_name"]
    device     = cfg["device"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config() if cfg["use_4bit"] else None,
        device_map=device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if cfg["use_4bit"]:
        # use_reentrant=False is required for torch.autograd.grad() in SMART
        try:
            model = prepare_model_for_kbit_training(
                model,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            # Older PEFT versions don't have gradient_checkpointing_kwargs
            model = prepare_model_for_kbit_training(model)
            # Patch the checkpointing modules to use non-reentrant mode
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing:
                    module.gradient_checkpointing_kwargs = {"use_reentrant": False}

    return model, tok


def load_for_training(cfg: dict):
    """Return (model_with_lora, tokenizer) ready for training."""
    model, tok = _load_base(cfg)
    lora_cfg = LoraConfig(
        r               = cfg["lora_rank"],
        lora_alpha      = cfg["lora_alpha"],
        lora_dropout    = cfg["lora_dropout"],
        target_modules  = cfg["lora_targets"],
        bias            = "none",
        task_type       = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


def load_from_checkpoint(cfg: dict, ckpt_dir: str):
    """Return (model_with_lora, tokenizer) loaded from saved adapter."""
    model, tok = _load_base(cfg)
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model.eval()
    print(f"  Loaded adapter: {ckpt_dir}")
    return model, tok


# ── Tokenisation helpers ───────────────────────────────────────────────────

def get_label_ids(tok, label_chars: list) -> list:
    """Return token IDs for label characters A/B/C/D."""
    ids = []
    for c in label_chars:
        toks = tok.encode(c, add_special_tokens=False)
        ids.append(toks[-1])
    return ids


def tokenise_batch(tok, prompts: list, label_chars: list,
                   max_len: int = 256) -> dict:
    """
    Left-pad prompt+label, return {input_ids, attention_mask, label_positions}.
    label_positions[i] = position of the label token (always the last token).
    """
    pad_id = tok.pad_token_id or tok.eos_token_id
    seqs = []
    for p, lc in zip(prompts, label_chars):
        p_ids = tok.encode(p, add_special_tokens=False)
        l_ids = tok.encode(lc, add_special_tokens=False)
        seq   = (p_ids + [l_ids[-1]])[-max_len:]
        seqs.append(seq)

    L = max(len(s) for s in seqs)
    input_ids, attn = [], []
    for s in seqs:
        pad = L - len(s)
        input_ids.append([pad_id] * pad + s)
        attn.append([0] * pad + [1] * len(s))

    return dict(
        input_ids      = torch.tensor(input_ids, dtype=torch.long),
        attention_mask = torch.tensor(attn,      dtype=torch.long),
        label_positions= torch.full((len(seqs),), L - 1, dtype=torch.long),
    )


def extract_class_logits(logits: torch.Tensor,
                         label_positions: torch.Tensor,
                         label_ids: list) -> torch.Tensor:
    """
    logits: (B, T, V) → (B, K)  restricted to label token positions.
    Position of the *prediction* token is label_positions - 1
    (the model predicts the label from the token before it).
    """
    B     = logits.shape[0]
    pos   = label_positions - 1          # (B,)
    pred  = logits[torch.arange(B, device=logits.device), pos]   # (B, V)
    ids_t = torch.tensor(label_ids, device=logits.device)
    return pred[:, ids_t]                # (B, K)


# ── Embedding layer access ─────────────────────────────────────────────────

def get_embed_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the token embedding layer from a (possibly PEFT-wrapped) model.
    Tries common paths in order of likelihood.
    """
    candidates = [
        "base_model.model.model.embed_tokens",   # PEFT + Qwen VL
        "base_model.model.embed_tokens",           # PEFT + standard LM
        "model.model.embed_tokens",                # unwrapped Qwen
        "model.embed_tokens",
        "transformer.wte",                         # GPT-style
    ]
    for path in candidates:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    raise ValueError("Cannot find embed_tokens layer in model")


def forward_from_embeds(model: torch.nn.Module,
                        inputs_embeds: torch.Tensor,
                        attention_mask: torch.Tensor,
                        label_positions: torch.Tensor,
                        label_ids: list) -> torch.Tensor:
    """
    Forward pass using pre-computed embeddings (for R3F / SMART).
    Returns (B, K) class logits.
    """
    out = model(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask)
    return extract_class_logits(out.logits, label_positions, label_ids)
