"""
Result aggregation, visualisation, and summary generation.
Reads results.csv → produces tables, plots, and summary.md.
"""
import os, sys, json, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR, TASK_CFGS

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_results(path=None):
    path = path or os.path.join(RESULTS_DIR, "results.csv")
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────

def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds, produce mean ± std for key metrics."""
    group_cols = ["task", "modality", "method", "perturbation"]
    metric_cols = ["accuracy", "worst_class_acc", "vwr_gamma", "sigma_max"]
    present = [c for c in metric_cols if c in df.columns]

    agg = df.groupby(group_cols)[present].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

METHOD_ORDER = ["base_clean", "base_aug", "plugin"]
METHOD_COLORS = {"base_clean": "#1f77b4", "base_aug": "#ff7f0e", "plugin": "#2ca02c"}


def plot_clean_vs_robust(df: pd.DataFrame, outdir=PLOT_DIR):
    """Bar chart: clean acc vs average robust acc per method/task."""
    for task in df["task"].unique():
        td = df[df["task"] == task]
        clean = td[td["perturbation"] == "clean"].groupby("method")["accuracy"].mean()
        robust = td[td["perturbation"] != "clean"].groupby("method")["accuracy"].mean()

        methods = [m for m in METHOD_ORDER if m in clean.index]
        x = np.arange(len(methods))
        w = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w/2, [clean.get(m, 0) for m in methods], w, label="Clean", color="#66b3ff")
        ax.bar(x + w/2, [robust.get(m, 0) for m in methods], w, label="Robust (avg)", color="#ff9999")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task}: Clean vs Robust Accuracy")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{task}_clean_vs_robust.png"), dpi=150)
        plt.close(fig)


def plot_worst_class(df: pd.DataFrame, outdir=PLOT_DIR):
    for task in df["task"].unique():
        td = df[(df["task"] == task) & (df["perturbation"] != "clean")]
        grp = td.groupby("method")["worst_class_acc"].mean()
        methods = [m for m in METHOD_ORDER if m in grp.index]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(methods, [grp.get(m, 0) for m in methods],
               color=[METHOD_COLORS.get(m, "gray") for m in methods])
        ax.set_ylabel("Worst-Class Robust Accuracy")
        ax.set_title(f"{task}: Worst-Class Comparison")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{task}_worst_class.png"), dpi=150)
        plt.close(fig)


def plot_vwr_sigma(df: pd.DataFrame, outdir=PLOT_DIR):
    for metric, label in [("vwr_gamma", "VWR_γ"), ("sigma_max", "σ_max")]:
        if metric not in df.columns:
            continue
        for task in df["task"].unique():
            td = df[(df["task"] == task) & (df["perturbation"] != "clean")]
            grp = td.groupby("method")[metric].mean()
            methods = [m for m in METHOD_ORDER if m in grp.index]

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(methods, [grp.get(m, 0) for m in methods],
                   color=[METHOD_COLORS.get(m, "gray") for m in methods])
            ax.set_ylabel(label)
            ax.set_title(f"{task}: {label} Comparison")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{task}_{metric}.png"), dpi=150)
            plt.close(fig)


def plot_per_perturbation(df: pd.DataFrame, outdir=PLOT_DIR):
    for task in df["task"].unique():
        td = df[df["task"] == task]
        ptypes = [p for p in td["perturbation"].unique() if p != "clean"]
        if not ptypes:
            continue
        methods = [m for m in METHOD_ORDER if m in td["method"].unique()]

        fig, ax = plt.subplots(figsize=(max(6, len(ptypes)*1.5), 4))
        x = np.arange(len(ptypes))
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            vals = []
            for p in ptypes:
                sub = td[(td["method"] == m) & (td["perturbation"] == p)]
                vals.append(sub["accuracy"].mean() if len(sub) > 0 else 0)
            ax.bar(x + i * w, vals, w, label=m,
                   color=METHOD_COLORS.get(m, "gray"))
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(ptypes, rotation=30, ha="right")
        ax.set_ylabel("Robust Accuracy")
        ax.set_title(f"{task}: Per-Perturbation Accuracy")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{task}_per_perturbation.png"), dpi=150)
        plt.close(fig)


def plot_confusion_heatmap(confusion_json: list, task: str, method: str,
                           perturbation: str, label_names: list, outdir=PLOT_DIR):
    """Plot a single confusion matrix as a heatmap."""
    cm = np.array(confusion_json)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names,
                yticklabels=label_names, cmap="YlOrRd", ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{task} | {method} | {perturbation}")
    fig.tight_layout()
    fname = f"{task}_{method}_{perturbation}_confusion.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame, outdir=RESULTS_DIR):
    """Write summary.md with tables, conclusions, next steps."""
    lines = ["# PAC-Robust Plugin Regulariser — Feasibility Study Results\n"]

    # Key finding overview
    lines.append("## 1. Overview\n")
    lines.append("This report summarises a small-scale feasibility experiment testing whether a "
                 "**gamma-aware plugin regulariser** can reduce worst-class fragile behaviour "
                 "across text classification, text reasoning, multimodal reasoning, and "
                 "a robot action-selection demo.\n")

    # Per-task summary tables
    lines.append("## 2. Results Tables\n")
    for task in df["task"].unique():
        td = df[df["task"] == task]
        lines.append(f"### {task}\n")

        pivot = td.groupby(["method", "perturbation"])[
            ["accuracy", "worst_class_acc", "vwr_gamma", "sigma_max"]
        ].mean().reset_index()
        lines.append(pivot.to_markdown(index=False))
        lines.append("\n")

    # Clean vs Robust drop
    lines.append("## 3. Clean → Robust Drop\n")
    for task in df["task"].unique():
        td = df[df["task"] == task]
        for method in METHOD_ORDER:
            md = td[td["method"] == method]
            clean_acc = md[md["perturbation"] == "clean"]["accuracy"].mean()
            rob_acc = md[md["perturbation"] != "clean"]["accuracy"].mean()
            drop = clean_acc - rob_acc
            lines.append(f"- **{task} / {method}**: clean={clean_acc:.3f}, "
                         f"robust(avg)={rob_acc:.3f}, drop={drop:.3f}")
    lines.append("\n")

    # Key conclusions
    lines.append("## 4. Key Conclusions\n")
    # Auto-generate from data
    for task in df["task"].unique():
        td = df[(df["task"] == task) & (df["perturbation"] != "clean")]
        aug_worst = td[td["method"] == "base_aug"]["worst_class_acc"].mean()
        plugin_worst = td[td["method"] == "plugin"]["worst_class_acc"].mean() if "plugin" in td["method"].values else float("nan")
        aug_vwr = td[td["method"] == "base_aug"]["vwr_gamma"].mean()
        plugin_vwr = td[td["method"] == "plugin"]["vwr_gamma"].mean() if "plugin" in td["method"].values else float("nan")

        improved = plugin_worst > aug_worst if not math.isnan(plugin_worst) else False
        lines.append(f"### {task}")
        lines.append(f"- Plugin worst-class acc: {plugin_worst:.4f} vs Base-aug: {aug_worst:.4f} "
                     f"({'↑ improved' if improved else '→ not improved'})")
        lines.append(f"- Plugin VWR_γ: {plugin_vwr:.4f} vs Base-aug: {aug_vwr:.4f}")
        lines.append("")

    # Plots references
    lines.append("## 5. Plots\n")
    lines.append("See `results/plots/` for:\n")
    lines.append("- Clean vs Robust accuracy\n- Worst-class metric\n"
                 "- VWR_γ and σ_max comparisons\n- Per-perturbation accuracy\n"
                 "- Confusion matrices (robot demo)\n")

    # Simplifications
    lines.append("## 6. Simplifications Made\n")
    lines.append("- Used Qwen3-VL-2B-Instruct for all tasks (VLM for text-only introduces minor overhead)\n")
    lines.append("- 4-bit quantisation + LoRA rank 8 (minimal footprint)\n")
    lines.append("- Small data subsets for fast iteration\n")
    lines.append("- Exact SVD for R_spec (feasible since K≤6 classes)\n")
    lines.append("- Synthetic images for robot demo (not real robot data)\n")

    # Next steps
    lines.append("\n## 7. Recommended Next Steps\n")
    lines.append("1. **Scale data**: increase to full datasets if signal is positive\n")
    lines.append("2. **Add baselines**: R3F, SMART, adversarial training\n")
    lines.append("3. **Stronger perturbations**: adversarial attacks, compositional perturbations\n")
    lines.append("4. **More seeds**: 3–5 seeds for statistical significance\n")
    lines.append("5. **Dedicated text model**: use a pure LLM for text-only tasks\n")

    summary_path = os.path.join(outdir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary written to {summary_path}")


def generate_all(csv_path=None, confusion_data=None):
    df = load_results(csv_path)
    print(f"Loaded {len(df)} result rows.")

    plot_clean_vs_robust(df)
    plot_worst_class(df)
    plot_vwr_sigma(df)
    plot_per_perturbation(df)

    # Robot confusion matrices (if available)
    if confusion_data:
        for entry in confusion_data:
            task_cfg = TASK_CFGS.get(entry["task"], {})
            label_names = task_cfg.get("label_names", entry.get("label_names", []))
            plot_confusion_heatmap(entry["confusion"], entry["task"],
                                   entry["method"], entry["perturbation"],
                                   label_names)

    generate_summary(df)
    print("All analysis complete.")


if __name__ == "__main__":
    generate_all()
