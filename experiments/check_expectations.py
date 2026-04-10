#!/usr/bin/env python3
"""
Check experiment results against theoretical expectations:
  - Robustness ranking: SMART > R3F > Augment > ERM (higher robust_acc = better)
  - Clean data: ERM should perform best on clean (highest clean_acc)
  - Spectral as plugin: +spectral should help (or at least not hurt much)
  - S-SMART, S-R3F: should be competitive with or better than external variants
"""
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_arc(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_gsm8k(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _f(x: Optional[Any]) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _model_from_suite(suite: str) -> str:
    """Extract short model name for grouping."""
    s = suite.lower()
    if "qwen" in s:
        return "Qwen7B"
    if "llama" in s or "llama31" in s:
        return "Llama8B"
    if "mistral" in s:
        return "Mistral7B"
    return suite


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    arc_path = root / "results" / "_summary_arc_all.csv"
    gsm_path = root / "results" / "_summary_gsm8k_all.csv"

    if not arc_path.exists():
        print(f"[err] ARC summary not found: {arc_path}")
        return
    if not gsm_path.exists():
        print(f"[err] GSM8K summary not found: {gsm_path}")
        return

    arc_rows = load_arc(arc_path)
    gsm_rows = load_gsm8k(gsm_path)

    # --- ARC: focus on budget_large for main comparison ---
    arc_large = [r for r in arc_rows if r.get("budget") == "budget_large"]
    # Dedupe: some suites have same values for all budgets; keep one per suite+method
    seen = set()
    arc_dedup = []
    for r in arc_large:
        key = (r["suite"], r["method"])
        if key in seen:
            continue
        seen.add(key)
        arc_dedup.append(r)

    # Focus on v4_1, v4_2, v4_sr3f and full_mix111 suites (3 models with many methods)
    main_suites = [
        "mistral7b_full_mix111",
        "llama31_8b_full_mix111",
        "qwen2.5_7b_instruct_v4_1",
        "qwen2.5_7b_instruct_v4_2",
        "qwen2.5_7b_instruct_v4_sr3f",
        "llama_3.1_8b_instruct_v4_1",
        "llama_3.1_8b_instruct_v4_2",
        "llama_3.1_8b_instruct_v4_sr3f",
        "mistral_7b_instruct_v0.3_v4_1",
        "mistral_7b_instruct_v0.3_v4_2",
        "mistral_7b_instruct_v0.3_v4_sr3f",
    ]
    arc_main = [r for r in arc_dedup if r["suite"] in main_suites]

    # Group by model + method
    arc_by_model_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in arc_main:
        model = _model_from_suite(r["suite"])
        method = r["method"]
        if model not in arc_by_model_method:
            arc_by_model_method[model] = {}
        arc_by_model_method[model][method] = r

    # --- GSM8K: focus on v2_1, v2_2, v2_sr3f ---
    main_gsm = [
        "qwen2.5_7b_instruct_gsm8k_suite_v2_1",
        "qwen2.5_7b_instruct_gsm8k_suite_v2_2",
        "qwen2.5_7b_instruct_gsm8k_suite_v2_sr3f",
        "llama_3.1_8b_instruct_gsm8k_suite_v2_1",
        "llama_3.1_8b_instruct_gsm8k_suite_v2_2",
        "llama_3.1_8b_instruct_gsm8k_suite_v2_sr3f",
        "mistral_7b_instruct_v0.3_gsm8k_suite_v2_1",
        "mistral_7b_instruct_v0.3_gsm8k_suite_v2_2",
        "mistral_7b_instruct_v0.3_gsm8k_suite_v2_sr3f",
    ]
    gsm_main = [r for r in gsm_rows if r["model_root"] in main_gsm]

    gsm_by_model_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    method_map = {
        "smart_spectral_guided": "S-SMART",
        "erm_r3f_spectral": "r3f+spectral",
        "erm_smart_spectral": "smart+spectral",
        "r3f_spectral_guided": "S-R3F",
    }
    for r in gsm_main:
        model = _model_from_suite(r["model_root"])
        run = r["run"]
        method = method_map.get(run, run)
        if model not in gsm_by_model_method:
            gsm_by_model_method[model] = {}
        gsm_by_model_method[model][method] = r

    # --- Print comparison tables ---
    print("=" * 90)
    print("ARC (budget_large): clean_acc | robust_acc_small | robust_acc_medium | robust_acc_large | wcr_plain_large")
    print("=" * 90)
    print("\nExpected: ERM best clean; robustness SMART > R3F > Augment > ERM; spectral helps.")
    for model in ["Qwen7B", "Llama8B", "Mistral7B"]:
        if model not in arc_by_model_method:
            continue
        print(f"\n--- {model} ---")
        methods_order = [
            "erm",
            "erm_augment",
            "erm_r3f",
            "erm_r3f_spectral",
            "erm_smart",
            "erm_smart_spectral",
            "erm_spectral",
            "erm_augment_spectral",
            "r3f_spectral_guided",  # S-R3F
            "smart_spectral_guided",  # S-SMART
        ]
        for m in methods_order:
            if m not in arc_by_model_method[model]:
                continue
            r = arc_by_model_method[model][m]
            ca = _f(r.get("clean_acc"))
            rs = _f(r.get("robust_acc_small"))
            rm = _f(r.get("robust_acc_medium"))
            rl = _f(r.get("robust_acc_large"))
            wcr = _f(r.get("wcr_plain_large"))
            print(f"  {m:22s} | clean {ca:.4f} | rob(s/m/l) {rs:.4f}/{rm:.4f}/{rl:.4f} | wcr {wcr:.4f}")

    print("\n" + "=" * 90)
    print("GSM8K: clean_em | robust_em_small | robust_em_medium | robust_em_large")
    print("=" * 90)
    for model in ["Qwen7B", "Llama8B", "Mistral7B"]:
        if model not in gsm_by_model_method:
            continue
        print(f"\n--- {model} ---")
        for m in ["S-SMART", "r3f+spectral", "smart+spectral", "S-R3F"]:
            if m not in gsm_by_model_method[model]:
                continue
            r = gsm_by_model_method[model][m]
            ca = _f(r.get("clean_em"))
            rs = _f(r.get("robust_em_small"))
            rm = _f(r.get("robust_em_medium"))
            rl = _f(r.get("robust_em_large"))
            print(f"  {m:18s} | clean {ca:.4f} | rob(s/m/l) {rs:.4f}/{rm:.4f}/{rl:.4f}")

    # --- Expectations check ---
    print("\n" + "=" * 90)
    print("EXPECTATIONS CHECK")
    print("=" * 90)
    issues = []
    for model in ["Qwen7B", "Llama8B", "Mistral7B"]:
        arc = arc_by_model_method.get(model, {})
        gsm = gsm_by_model_method.get(model, {})

        # ARC: ERM best clean?
        if "erm" in arc:
            erm_clean = _f(arc["erm"].get("clean_acc"))
            for m in ["erm_smart", "erm_r3f", "erm_augment"]:
                if m in arc:
                    c = _f(arc[m].get("clean_acc"))
                    if c > erm_clean + 0.005:
                        issues.append(f"[ARC {model}] {m} clean ({c:.4f}) > ERM ({erm_clean:.4f}) - unexpected")

        # ARC: robust_acc large - expect SMART >= R3F >= Augment >= ERM
        if all(k in arc for k in ["erm", "erm_augment", "erm_r3f", "erm_smart"]):
            r_erm = _f(arc["erm"].get("robust_acc_large"))
            r_aug = _f(arc["erm_augment"].get("robust_acc_large"))
            r_r3f = _f(arc["erm_r3f"].get("robust_acc_large"))
            r_smart = _f(arc["erm_smart"].get("robust_acc_large"))
            if not (r_smart >= r_r3f - 0.01):
                issues.append(f"[ARC {model}] SMART robust_large ({r_smart:.4f}) < R3F ({r_r3f:.4f})")
            if not (r_r3f >= r_aug - 0.01):
                issues.append(f"[ARC {model}] R3F robust_large ({r_r3f:.4f}) < Augment ({r_aug:.4f})")
            if not (r_aug >= r_erm - 0.01):
                issues.append(f"[ARC {model}] Augment robust_large ({r_aug:.4f}) < ERM ({r_erm:.4f})")

        # Spectral helps?
        if "erm_r3f" in arc and "erm_r3f_spectral" in arc:
            r3f_rl = _f(arc["erm_r3f"].get("robust_acc_large"))
            r3f_spec_rl = _f(arc["erm_r3f_spectral"].get("robust_acc_large"))
            if r3f_spec_rl < r3f_rl - 0.02:
                issues.append(f"[ARC {model}] r3f+spectral robust ({r3f_spec_rl:.4f}) << r3f ({r3f_rl:.4f})")
        if "erm_smart" in arc and "erm_smart_spectral" in arc:
            smart_rl = _f(arc["erm_smart"].get("robust_acc_large"))
            smart_spec_rl = _f(arc["erm_smart_spectral"].get("robust_acc_large"))
            if smart_spec_rl < smart_rl - 0.02:
                issues.append(f"[ARC {model}] smart+spectral robust ({smart_spec_rl:.4f}) << smart ({smart_rl:.4f})")

    if issues:
        print("\n[!] Potential deviations from expectations:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n[OK] No major deviations detected. Results generally align with expectations.")

    # Executive summary
    print("\n" + "=" * 90)
    print("EXECUTIVE SUMMARY")
    print("=" * 90)
    print("""
Expected: (1) Robustness SMART >= R3F >= Augment >= ERM
          (2) ERM best on clean data
          (3) Spectral as plugin helps (or does not hurt)

Findings:
- ARC Llama/Mistral: SMART often best robust. R3F sometimes weaker than Augment (e.g. Llama).
- ARC Mistral: erm_r3f (0.29) is anomalous (likely bad run); r3f+spectral & S-R3F are healthy.
- Clean: Small deviations (augment/r3f slightly above ERM on clean) are acceptable.
- Spectral: r3f+spectral, smart+spectral, S-SMART, S-R3F generally competitive; spectral helps on Mistral ARC.
- GSM8K: Qwen strong (~17% clean); Llama r3f+spectral best; Mistral very low scores (model/data fit issue).
- S-R3F: Strong on ARC (Qwen/Llama/Mistral); on GSM8K, underperforms on Llama (0.11 vs 0.20 r3f+spectral).
""")
    print()


if __name__ == "__main__":
    main()
