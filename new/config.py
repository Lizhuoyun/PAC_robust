"""
Experiment configuration for PAC-robust plugin regularizer feasibility study.
All key hyperparameters are defined here—nothing is hardcoded elsewhere.
"""
import os, copy

HF_CACHE = "/LOCAL2/zhuoyun/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")

BASE_DIR = "/LOCAL2/zhuoyun/PAC_robust/new"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

DEFAULT_CFG = dict(
    model_name=MODEL_NAME,
    hf_cache=HF_CACHE,
    output_dir=RESULTS_DIR,
    checkpoint_dir=CHECKPOINT_DIR,

    # LoRA
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "v_proj"],

    # Quantization
    use_4bit=True,

    # Training
    lr=2e-4,
    num_epochs=3,
    batch_size=4,
    grad_accum_steps=4,
    warmup_ratio=0.1,
    max_seq_len=384,
    weight_decay=0.01,

    # Plugin
    alpha=0.1,
    beta=0.05,
    kappa=0.5,
    gamma=None,  # calibrated from Base-aug margins
    gamma_quantiles=[0.10, 0.25, 0.50],
    default_gamma_quantile=0.25,

    # Sweeps
    alpha_values=[0.05, 0.1, 0.2],
    beta_values=[0.0, 0.05],

    # Seeds
    seeds=[42, 123],

    # Perturbation
    text_perturbation_types=["typo", "distractor", "format_rewrite"],
    image_perturbation_types=["blur", "jpeg", "resize"],
    joint_perturbation_types=["joint"],
    typo_rate=0.10,
    jpeg_quality=20,
    blur_kernel=5,
    resize_scale=0.25,

    # Device
    device="cuda:0",
)

TASK_CFGS = dict(
    agnews=dict(
        task_name="agnews",
        dataset_name="ag_news",
        modality="text",
        num_classes=4,
        label_chars=["A", "B", "C", "D"],
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        train_size=2000,
        val_size=500,
        test_size=500,
        perturbation_types=["typo", "distractor", "format_rewrite"],
    ),
    arc=dict(
        task_name="arc",
        dataset_name="allenai/ai2_arc",
        dataset_subset="ARC-Challenge",
        modality="text",
        num_classes=4,
        label_chars=["A", "B", "C", "D"],
        label_names=["A", "B", "C", "D"],
        train_size=1000,
        val_size=295,
        test_size=500,
        perturbation_types=["typo", "distractor", "format_rewrite"],
    ),
    scienceqa=dict(
        task_name="scienceqa",
        dataset_name="derek-thomas/ScienceQA",
        modality="multimodal",
        num_classes=4,
        label_chars=["A", "B", "C", "D"],
        label_names=["A", "B", "C", "D"],
        train_size=1500,
        val_size=400,
        test_size=400,
        perturbation_types=["typo", "distractor", "format_rewrite",
                            "blur", "jpeg", "resize", "joint"],
    ),
    robot=dict(
        task_name="robot",
        modality="multimodal",
        num_classes=6,
        label_chars=["A", "B", "C", "D", "E", "F"],
        label_names=["pick", "push", "place", "move_left", "move_right", "stop"],
        train_size=1500,
        val_size=300,
        test_size=300,
        perturbation_types=["typo", "distractor",
                            "blur", "jpeg", "resize", "joint"],
    ),
)


def get_cfg(task_name: str, **overrides) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.update(copy.deepcopy(TASK_CFGS[task_name]))
    cfg.update(overrides)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    return cfg
