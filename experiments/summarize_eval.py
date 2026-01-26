#!/usr/bin/env python3
"""Summarize evaluation results from qwen7b_tuned_v3 and compare with previous versions."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

def load_all_metrics(root_dir: str) -> dict:
    """Load all eval/metrics.json files and organize by method_budget."""
    results = defaultdict(dict)
    root_path = Path(root_dir)
    for metrics_file in root_path.rglob("eval/metrics.json"):
        # Parse path: results/arc/{suite}/{method}/{budget}/seed42/eval/metrics.json
        parts = metrics_file.parts
        try:
            # Find method and budget indices
            method_idx = None
            budget_idx = None
            for i, part in enumerate(parts):
                if part in ("erm", "erm_spectral", "erm_r3f", "erm_r3f_spectral", 
                           "erm_smart", "erm_smart_spectral", "erm_augment", "erm_augment_spectral"):
                    method_idx = i
                elif part.startswith("budget_"):
                    budget_idx = i
            if method_idx is None or budget_idx is None:
                continue
            method = parts[method_idx]
            budget = parts[budget_idx]
            key = f"{method}_{budget}"
            with open(metrics_file) as f:
                data = json.load(f)
            results[key] = data
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}", file=sys.stderr)
            continue
    return results

def print_summary(results: dict, title: str = ""):
    """Print a formatted summary table."""
    if title:
        print(f"\n{'='*150}")
        print(f"{title:^150}")
        print(f"{'='*150}")
    
    # Header
    header = "Method_Budget".ljust(35) + " | clean_acc | robust_acc_small | robust_acc_medium | robust_acc_large | wcr_small | wcr_medium | wcr_large | sigma_max_small | sigma_max_medium | sigma_max_large"
    print(header)
    print("-" * len(header))
    
    # Sort by method, then budget
    def sort_key(k):
        method_order = {"erm": 0, "erm_spectral": 1, "erm_r3f": 2, "erm_r3f_spectral": 3,
                       "erm_smart": 4, "erm_smart_spectral": 5, "erm_augment": 6, "erm_augment_spectral": 7}
        budget_order = {"budget_small": 0, "budget_medium": 1, "budget_large": 2}
        method = k.split("_budget_")[0]
        budget = k.split("_budget_")[1] if "_budget_" in k else k.split("budget_")[1]
        return (method_order.get(method, 999), budget_order.get(budget, 999))
    
    for key in sorted(results.keys(), key=sort_key):
        m = results[key]
        row = (
            f"{key:35s} | "
            f"{m.get('clean_acc', 0):.4f} | "
            f"{m.get('robust_acc_small', 0):.4f} | "
            f"{m.get('robust_acc_medium', 0):.4f} | "
            f"{m.get('robust_acc_large', 0):.4f} | "
            f"{m.get('wcr_small', 0):.4f} | "
            f"{m.get('wcr_medium', 0):.4f} | "
            f"{m.get('wcr_large', 0):.4f} | "
            f"{m.get('sigma_max_small', 0):.4f} | "
            f"{m.get('sigma_max_medium', 0):.4f} | "
            f"{m.get('sigma_max_large', 0):.4f}"
        )
        print(row)

def compare_results(v3: dict, baseline: dict, baseline_name: str = "baseline"):
    """Compare v3 results with baseline and show deltas."""
    print(f"\n{'='*150}")
    print(f"Delta Comparison: qwen7b_tuned_v3 vs {baseline_name}")
    print(f"{'='*150}")
    
    header = "Method_Budget".ljust(35) + " | clean_acc_delta | robust_acc_small_delta | robust_acc_medium_delta | robust_acc_large_delta | wcr_small_delta | wcr_medium_delta | wcr_large_delta"
    print(header)
    print("-" * len(header))
    
    all_keys = sorted(set(v3.keys()) | set(baseline.keys()))
    for key in all_keys:
        v3_data = v3.get(key, {})
        base_data = baseline.get(key, {})
        
        def delta(k):
            v3_val = v3_data.get(k, 0)
            base_val = base_data.get(k, 0)
            return v3_val - base_val
        
        row = (
            f"{key:35s} | "
            f"{delta('clean_acc'):+.4f} | "
            f"{delta('robust_acc_small'):+.4f} | "
            f"{delta('robust_acc_medium'):+.4f} | "
            f"{delta('robust_acc_large'):+.4f} | "
            f"{delta('wcr_small'):+.4f} | "
            f"{delta('wcr_medium'):+.4f} | "
            f"{delta('wcr_large'):+.4f}"
        )
        print(row)

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    
    # Load v3 results
    v3_dir = base_dir / "results" / "arc" / "qwen7b_tuned_v3"
    v3_results = load_all_metrics(str(v3_dir))
    print_summary(v3_results, "qwen7b_tuned_v3 Evaluation Results")
    
    # Try to load baseline for comparison
    baseline_dirs = [
        ("qwen7b_tuned_v2_fix", "qwen7b_tuned_v2_fix"),
        ("qwen7b_full_mix111", "qwen7b_full_mix111"),
    ]
    
    for dirname, label in baseline_dirs:
        baseline_dir = base_dir / "results" / "arc" / dirname
        if baseline_dir.exists():
            baseline_results = load_all_metrics(str(baseline_dir))
            if baseline_results:
                compare_results(v3_results, baseline_results, label)



