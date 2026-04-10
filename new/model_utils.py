"""
Model / tokeniser / processor loading with LoRA + 4-bit quantisation.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training


def _load_base_model(cfg: dict):
    """Load base model + tokenizer + processor (without LoRA)."""
    model_name = cfg["model_name"]
    device = cfg["device"]
    use_4bit = cfg.get("use_4bit", True)

    bnb_cfg = None
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    is_vlm = "VL" in model_name or "vl" in model_name

    if is_vlm:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device,
            torch_dtype=torch.bfloat16,
            cache_dir=cfg["hf_cache"],
        )
        processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cfg["hf_cache"],
        )
        tokenizer = processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device,
            torch_dtype=torch.bfloat16,
            cache_dir=cfg["hf_cache"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cfg["hf_cache"],
        )
        processor = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer, processor


def load_model_and_tokenizer(cfg: dict):
    """Load base model with fresh LoRA adapters (for training)."""
    model, tokenizer, processor = _load_base_model(cfg)

    lora_cfg = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer, processor


def load_model_from_checkpoint(cfg: dict, checkpoint_dir: str):
    """Load base model + saved LoRA adapter (for evaluation)."""
    model, tokenizer, processor = _load_base_model(cfg)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    print(f"Loaded adapter from {checkpoint_dir}")
    return model, tokenizer, processor


def get_label_token_ids(tokenizer, label_chars: list) -> list:
    """Get token IDs for label characters (A, B, C, D, ...)."""
    ids = []
    for c in label_chars:
        toks = tokenizer.encode(c, add_special_tokens=False)
        ids.append(toks[-1])
    return ids


def tokenize_for_classification(tokenizer, prompts: list, label_chars: list,
                                max_len: int = 384) -> dict:
    """
    Tokenise prompt+label with left-padding.  Label is always the last token.
    Returns dict(input_ids, attention_mask, label_positions) — all tensors.
    """
    import torch
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    all_ids = []
    for prompt, lc in zip(prompts, label_chars):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        l_ids = tokenizer.encode(lc, add_special_tokens=False)
        seq = p_ids + [l_ids[-1]]
        if len(seq) > max_len:
            seq = seq[-(max_len):]
        all_ids.append(seq)

    max_len_batch = max(len(s) for s in all_ids)
    input_ids, attn = [], []
    for seq in all_ids:
        pad_len = max_len_batch - len(seq)
        input_ids.append([pad_id] * pad_len + seq)
        attn.append([0] * pad_len + [1] * len(seq))

    return dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attn, dtype=torch.long),
        label_positions=torch.full((len(prompts),), max_len_batch - 1, dtype=torch.long),
    )


def extract_class_logits(logits: torch.Tensor, label_positions: torch.Tensor,
                         label_token_ids: list) -> torch.Tensor:
    """
    logits: (B, T, V)
    Returns: (B, num_classes) — logits restricted to label tokens at predict positions.
    """
    B = logits.shape[0]
    pred_pos = label_positions - 1
    pred_logits = logits[torch.arange(B, device=logits.device), pred_pos]  # (B, V)
    label_ids_t = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    return pred_logits[:, label_ids_t]  # (B, K)
