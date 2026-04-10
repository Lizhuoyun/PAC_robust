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

# ── Model ──────────────────────────────────────────────────────────────────
# Qwen2.5-0.5B-Instruct — pure text causal LM, local snapshot path
MODEL_NAME = (
    "/LOCAL2/zhuoyun/hf_cache/models/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
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
# Priority 1 (must run)
P1_METHODS = [
    "base_clean", "base_aug", "plugin",
    "r3f", "r3f_plugin",
    "smart", "smart_plugin",
]
# Priority 2 (optional)
P2_METHODS = ["awp", "awp_plugin"]


def get_cfg(**overrides) -> dict:
    """Return a config dict, optionally overriding defaults."""
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
    cfg.update(overrides)
    return cfg
