"""
Global configuration for PAC-robust v2.
All hyperparameters in one place — never hardcode elsewhere.
"""
import os

HF_CACHE = "/LOCAL2/zhuoyun/hf_cache"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE, "hub"))

BASE_DIR    = "/LOCAL2/zhuoyun/PAC_robust/v2"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")


def _resolve_snapshot(model_dir: str, preferred_hash: str = None,
                      env_var: str = None, required: bool = True) -> str:
    """
    Resolve a local snapshot path from HF cache or an explicit env override.

    If `preferred_hash` is unavailable but snapshots exist, fall back to the
    lexicographically last snapshot directory so newer downloads work without
    editing the repo. Optional models return None when absent.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            if os.path.isdir(override):
                return override
            if required:
                raise FileNotFoundError(
                    f"{env_var} points to a missing directory: {override}")
            return None

    snapshots_dir = os.path.join(HF_CACHE, "models", model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        if required:
            raise FileNotFoundError(f"Missing snapshots directory: {snapshots_dir}")
        return None

    if preferred_hash:
        preferred = os.path.join(snapshots_dir, preferred_hash)
        if os.path.isdir(preferred):
            return preferred

    candidates = sorted(
        name for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    )
    if not candidates:
        if required:
            raise FileNotFoundError(f"No snapshot folders found in: {snapshots_dir}")
        return None
    return os.path.join(snapshots_dir, candidates[-1])

# ── Model ──────────────────────────────────────────────────────────────────
# Qwen2.5-0.5B-Instruct — pure text causal LM, local snapshot path
MODEL_NAME = _resolve_snapshot(
    "models--Qwen--Qwen2.5-0.5B-Instruct",
    preferred_hash="7ae557604adf67be50417f59c2c2f167def9a775",
    env_var="QWEN05B_SNAPSHOT",
)

# ── LoRA ───────────────────────────────────────────────────────────────────
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ── Training ───────────────────────────────────────────────────────────────
# Batch=8 keeps each job ~3-4 GB so multiple jobs fit per GPU
LR             = 2e-4
NUM_EPOCHS     = 5
BATCH_SIZE     = 8
GRAD_ACCUM     = 4      # effective batch = 32
WARMUP_RATIO   = 0.10
MAX_SEQ_LEN    = 256
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0

# ── ARC-Challenge task ─────────────────────────────────────────────────────
TASK        = "arc"
NUM_CLASSES = 4
LABEL_CHARS = ["A", "B", "C", "D"]
TRAIN_SIZE  = 1000
VAL_SIZE    = 295       # use full ARC val set
TEST_SIZE   = 500

# ── Perturbations ──────────────────────────────────────────────────────────
PERTURB_TYPES = ["typo", "distractor", "format_rewrite"]
TYPO_RATE     = 0.10

# ── Plugin ─────────────────────────────────────────────────────────────────
ALPHA           = 0.10    # R_spec weight
BETA            = 0.05    # R_stab weight
KAPPA           = 0.50    # gate temperature
GAMMA_QUANTILES = [0.10, 0.25, 0.50]
DEFAULT_GAMMA_Q = 0.25

# ── R3F ────────────────────────────────────────────────────────────────────
R3F_LAMBDA    = 1.0
R3F_NOISE_STD = 1e-3

# ── SMART ──────────────────────────────────────────────────────────────────
SMART_LAMBDA    = 1.0
SMART_EPSILON   = 1e-3
SMART_STEP_SIZE = 1e-3
SMART_STEPS     = 1       # PGD steps; increase to 3 for stronger SMART

# ── AWP ────────────────────────────────────────────────────────────────────
AWP_LAMBDA  = 1.0
AWP_ADV_LR  = 1e-4
AWP_ADV_EPS = 1e-4

# ── Experiment ─────────────────────────────────────────────────────────────
SEEDS   = [42, 123]
DEVICE  = "cuda:0"

# ── All method names ───────────────────────────────────────────────────────
ALL_METHODS = [
    "base_clean",
    "base_aug",
    "plugin",
    "r3f",
    "r3f_plugin",
    "smart",
    "smart_plugin",
    "awp",
    "awp_plugin",
]
P1_METHODS = [
    "base_clean", "base_aug", "plugin",
    "r3f", "r3f_plugin",
    "smart", "smart_plugin",
]
P2_METHODS = ["awp", "awp_plugin"]

# ── Model registry (local snapshot paths) ─────────────────────────────────
MODEL_REGISTRY = {
    "qwen05b": _resolve_snapshot(
        "models--Qwen--Qwen2.5-0.5B-Instruct",
        preferred_hash="7ae557604adf67be50417f59c2c2f167def9a775",
        env_var="QWEN05B_SNAPSHOT",
    ),
    "qwen7b": _resolve_snapshot(
        "models--Qwen--Qwen2.5-7B-Instruct",
        preferred_hash="a09a35458c702b33eeacc393d103063234e8bc28",
        env_var="QWEN7B_SNAPSHOT",
    ),
    "mistral7b": _resolve_snapshot(
        "models--mistralai--Mistral-7B-Instruct-v0.2",
        preferred_hash="3ad372fc79158a2148299e3318516c786aeded6c",
        env_var="MISTRAL7B_SNAPSHOT",
    ),
}

_OPTIONAL_MODELS = {
    "qwen15b": _resolve_snapshot(
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        env_var="QWEN15B_SNAPSHOT",
        required=False,
    ),
    "qwen3b": _resolve_snapshot(
        "models--Qwen--Qwen2.5-3B-Instruct",
        env_var="QWEN3B_SNAPSHOT",
        required=False,
    ),
}

for _tag, _path in _OPTIONAL_MODELS.items():
    if _path:
        MODEL_REGISTRY[_tag] = _path

# ── Task presets ──────────────────────────────────────────────────────────
TASK_PRESETS = {
    "arc": dict(
        task_name   = "arc",
        num_classes = 4,
        label_chars = ["A", "B", "C", "D"],
        train_size  = 1000,
        val_size    = 295,
        test_size   = 500,
    ),
    "boolq": dict(
        task_name   = "boolq",
        num_classes = 2,
        label_chars = ["A", "B"],
        train_size  = 1000,
        val_size    = 500,
        test_size   = 500,
        max_seq_len = 384,
    ),
}


def get_cfg(**overrides) -> dict:
    """Return a config dict, optionally overriding defaults.
    
    If 'task_name' is in overrides, auto-applies TASK_PRESETS for that task
    *before* user overrides, so user values still win.
    """
    cfg = dict(
        # ── identity ────────────────────────────────
        task_name   = TASK,
        model_name  = MODEL_NAME,
        hf_cache    = HF_CACHE,
        results_dir = RESULTS_DIR,
        ckpt_dir    = CKPT_DIR,
        device      = DEVICE,
        # ── data ────────────────────────────────────
        num_classes = NUM_CLASSES,
        label_chars = LABEL_CHARS,
        train_size  = TRAIN_SIZE,
        val_size    = VAL_SIZE,
        test_size   = TEST_SIZE,
        # ── model ───────────────────────────────────
        use_4bit         = True,
        lora_rank        = LORA_RANK,
        lora_alpha       = LORA_ALPHA,
        lora_dropout     = LORA_DROPOUT,
        lora_targets     = LORA_TARGETS,
        # ── training ────────────────────────────────
        lr             = LR,
        num_epochs     = NUM_EPOCHS,
        batch_size     = BATCH_SIZE,
        grad_accum     = GRAD_ACCUM,
        warmup_ratio   = WARMUP_RATIO,
        max_seq_len    = MAX_SEQ_LEN,
        weight_decay   = WEIGHT_DECAY,
        max_grad_norm  = MAX_GRAD_NORM,
        # ── perturbations ───────────────────────────
        perturb_types = PERTURB_TYPES,
        typo_rate     = TYPO_RATE,
        # ── plugin ──────────────────────────────────
        alpha           = ALPHA,
        beta            = BETA,
        kappa           = KAPPA,
        gamma_quantiles = GAMMA_QUANTILES,
        default_gamma_q = DEFAULT_GAMMA_Q,
        # ── r3f ─────────────────────────────────────
        r3f_lambda    = R3F_LAMBDA,
        r3f_noise_std = R3F_NOISE_STD,
        # ── smart ───────────────────────────────────
        smart_lambda    = SMART_LAMBDA,
        smart_epsilon   = SMART_EPSILON,
        smart_step_size = SMART_STEP_SIZE,
        smart_steps     = SMART_STEPS,
        # ── awp ─────────────────────────────────────
        awp_lambda  = AWP_LAMBDA,
        awp_adv_lr  = AWP_ADV_LR,
        awp_adv_eps = AWP_ADV_EPS,
        # ── experiment ──────────────────────────────
        seeds   = SEEDS,
    )
    task_key = overrides.get("task_name", cfg["task_name"])
    if task_key in TASK_PRESETS:
        cfg.update(TASK_PRESETS[task_key])
    cfg.update(overrides)
    return cfg
