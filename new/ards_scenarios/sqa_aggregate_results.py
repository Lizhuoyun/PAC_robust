"""
Aggregate all evaluation results into summary tables and CSV.
"""
import os, sys, json, csv
sys.path.insert(0, "/LOCAL2/zhuoyun/PAC_robust/new")

RESULTS_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/results"
GAMMA_FILE = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/gamma_calibration.json"


def load_all_results():
    results = []
    gamma_cal = {}
    if os.path.exists(GAMMA_FILE):
        gamma_cal = json.load(open(GAMMA_FILE))

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith("eval_") or not fname.endswith(".json"):
            continue
        method = fname.replace("eval_", "").replace(".json", "")
        data = json.load(open(os.path.join(RESULTS_DIR, fname)))

        gamma_train = 0.0
        if "plugin_q10" in method:
            gamma_train = gamma_cal.get("q10", 0)
        elif "plugin_q25" in method:
            gamma_train = gamma_cal.get("q25", 0)
        elif "plugin_q50" in method:
            gamma_train = gamma_cal.get("q50", 0)

        for etype, metrics in data.items():
            row = {
                "method": method,
                "gamma_train": gamma_train,
                "eval_type": etype,
                "accuracy": metrics["accuracy"],
                "worst_class_acc": metrics["worst_class_acc"],
                "worst_class_err": metrics["worst_class_err"],
                "vwr_gamma": metrics["vwr_gamma"],
                "sigma_max": metrics["sigma_max"],
                "avg_gate": metrics.get("avg_gate", 0),
                "fragile_ratio": metrics.get("fragile_ratio", 0),
                "gamma_eval": metrics.get("gamma", 0),
            }
            results.append(row)
    return results


def write_csv(results, path):
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)


def write_summary_md(results, path):
    gamma_cal = {}
    if os.path.exists(GAMMA_FILE):
        gamma_cal = json.load(open(GAMMA_FILE))

    with open(path, "w") as f:
        f.write("# ScienceQA Plugin-LoRA Experiment Results\n\n")

        f.write("## Gamma Calibration\n\n")
        if gamma_cal:
            for k, v in gamma_cal.items():
                f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Main Results Table\n\n")
        f.write("| Method | Gamma | Eval | Acc | Worst-Cls Acc | Worst-Cls Err | VWR_γ | σ_max | Avg Gate | Fragile % |\n")
        f.write("|--------|-------|------|-----|---------------|---------------|-------|-------|----------|----------|\n")
        for r in results:
            f.write(f"| {r['method']} | {r['gamma_train']:.4f} | {r['eval_type']} | "
                    f"{r['accuracy']:.4f} | {r['worst_class_acc']:.4f} | {r['worst_class_err']:.4f} | "
                    f"{r['vwr_gamma']:.4f} | {r['sigma_max']:.4f} | "
                    f"{r['avg_gate']:.4f} | {r['fragile_ratio']:.4f} |\n")
        f.write("\n")

        f.write("## LoRA vs Plugin Comparison\n\n")
        lora_results = {r["eval_type"]: r for r in results if r["method"] == "lora"}
        for r in results:
            if r["method"] == "lora":
                continue
            etype = r["eval_type"]
            if etype in lora_results:
                lr = lora_results[etype]
                f.write(f"### {r['method']} vs LoRA ({etype})\n\n")
                f.write(f"- Δ Accuracy: {r['accuracy'] - lr['accuracy']:+.4f}\n")
                f.write(f"- Δ Worst-Class Acc: {r['worst_class_acc'] - lr['worst_class_acc']:+.4f}\n")
                f.write(f"- Δ VWR_γ: {r['vwr_gamma'] - lr['vwr_gamma']:+.4f}\n")
                f.write(f"- Δ σ_max: {r['sigma_max'] - lr['sigma_max']:+.4f}\n\n")

        f.write("## Gamma Sweep Data\n\n")
        f.write("For plotting gamma sweep curves:\n\n")
        f.write("| Gamma (quantile) | Gamma Value | Clean Acc | SA Acc | PA Acc | Clean VWR | SA VWR | Clean σ_max | SA σ_max |\n")
        f.write("|------------------|-------------|-----------|--------|--------|-----------|--------|-------------|----------|\n")

        for method_prefix in ["plugin_q10", "plugin_q25", "plugin_q50"]:
            method_results = {r["eval_type"]: r for r in results if r["method"] == method_prefix}
            if not method_results:
                continue
            qname = method_prefix.replace("plugin_", "")
            gval = gamma_cal.get(qname, 0)
            clean_r = method_results.get("clean", {})
            sa_r = method_results.get("sa", {})
            pa_r = method_results.get("pa", {})
            f.write(f"| {qname} | {gval:.4f} | "
                    f"{clean_r.get('accuracy', 0):.4f} | "
                    f"{sa_r.get('accuracy', 0):.4f} | "
                    f"{pa_r.get('accuracy', 0):.4f} | "
                    f"{clean_r.get('vwr_gamma', 0):.4f} | "
                    f"{sa_r.get('vwr_gamma', 0):.4f} | "
                    f"{clean_r.get('sigma_max', 0):.4f} | "
                    f"{sa_r.get('sigma_max', 0):.4f} |\n")
        f.write("\n")


if __name__ == "__main__":
    results = load_all_results()
    if not results:
        print("No results found yet.")
    else:
        csv_path = os.path.join(RESULTS_DIR, "scienceqa_plugin_results.csv")
        md_path = os.path.join(RESULTS_DIR, "scienceqa_plugin_results_summary.md")
        write_csv(results, csv_path)
        write_summary_md(results, md_path)
        print(f"CSV saved to {csv_path}")
        print(f"Summary saved to {md_path}")
        print(f"\nTotal rows: {len(results)}")
        for r in results:
            print(f"  {r['method']:15s} {r['eval_type']:6s} "
                  f"acc={r['accuracy']:.4f} worst={r['worst_class_acc']:.4f} "
                  f"vwr={r['vwr_gamma']:.4f} σ_max={r['sigma_max']:.4f}")
