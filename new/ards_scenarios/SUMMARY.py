"""
=== REUSABLE BASELINES SUMMARY ===

Model: liuhaotian/llava-v1.5-7b (pre-trained, zero-shot baseline)
Environment: /LOCAL2/zhuoyun/PAC_robust/ards_venv
ARDS repo: /LOCAL2/zhuoyun/PAC_robust/ARDS
"""

RESULTS = {
    "ScienceQA Clean":   {"acc": 70.22, "correct": 2978, "count": 4241, "status": "DONE"},
    "ScienceQA SA":      {"acc": 49.61, "correct": 2104, "count": 4241, "status": "DONE"},
    "ScienceQA PA":      {"acc": 46.26, "correct": 1962, "count": 4241, "status": "DONE"},
    "SEED-Bench Clean":  {"acc": 66.20, "correct": 9422, "count": 14233, "status": "DONE"},
    "SEED-Bench SA":     {"acc": 58.13, "correct": 8274, "count": 14233, "status": "DONE"},
}

SEED_CLEAN_PER_TYPE = {
    1: {"name": "Scene Understanding",    "acc": 74.0, "correct": 2338, "count": 3158},
    2: {"name": "Instance Identity",      "acc": 68.9, "correct": 1262, "count": 1831},
    3: {"name": "Instance Attributes",    "acc": 67.0, "correct": 3112, "count": 4648},
    4: {"name": "Instance Location",      "acc": 59.9, "correct": 586,  "count": 978},
    5: {"name": "Instance Counting",      "acc": 58.6, "correct": 1433, "count": 2447},
    6: {"name": "Spatial Relation",       "acc": 51.4, "correct": 338,  "count": 657},
    7: {"name": "Instance Interaction",   "acc": 69.1, "correct": 67,   "count": 97},
    8: {"name": "Visual Reasoning",       "acc": 76.7, "correct": 254,  "count": 331},
    9: {"name": "Text Understanding",     "acc": 37.2, "correct": 32,   "count": 86},
}

OUTPUT_PATHS = {
    "venv":           "/LOCAL2/zhuoyun/PAC_robust/ards_venv",
    "model":          "/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b",
    "sqa_clean":      "eval_outputs/scienceqa/clean/llava-v1.5-7b.jsonl",
    "sqa_sa":         "eval_outputs/scienceqa/sa_attack/llava-v1.5-7b.jsonl",
    "sqa_pa":         "eval_outputs/scienceqa/pa_attack/llava-v1.5-7b.jsonl",
    "seed_clean":     "eval_outputs/seed_bench/clean/llava-v1.5-7b/merge.jsonl",
    "seed_sa":        "eval_outputs/seed_bench/sa_attack/llava-v1.5-7b/merge.jsonl",
    "all_results":    "eval_outputs/all_results.json",
}

FIXES = [
    "model_vqa_science_option_attack.py: Added missing --eval-img argument",
    "ScienceQA SA: Generated llava_test_CQM-A_convertedABCDE-QWERT.json (ABCDE->QWERT)",
    "SEED-Bench: Fixed image paths (no .jpg extension), generated QWER-converted files",
    "Environment: Created dedicated venv (transformers==4.37.2 required by ARDS)",
]

PENDING = [
    "SEED-Bench SA accuracy: run compute_seed_sa.py (inference already done)",
    "SEED-Bench PA: not yet run",
    "MMBench: model_vqa_mmbench.py missing from ARDS repo",
]
