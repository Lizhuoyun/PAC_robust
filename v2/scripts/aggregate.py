#!/usr/bin/env python3
"""
Aggregate results across seeds and generate paper-ready tables.

Outputs:
  results/results_combined.csv   — all raw rows merged
  results/table1_main.csv        — mean±std per (method, perturbation)
  results/table2_plugin_gain.csv — plugin delta table
  results/table3_ablation.csv    — ablation (if ablation runs exist)
  results/gamma_sweep_plot_data.json — gamma sweep curves

Usage:
  python scripts/aggregate.py [--gamma_q 0.25]
"""
import sys, os, glob, csv, json, argparse
from collections import defaultdict
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, DEFAULT_GAMMA_Q


METRICS = ["accuracy", "worst_class_acc", "worst_class_err",
           "vwr_gamma", "sigma_max", "clean_to_robust_drop",
           "fragile_ratio", "mean_gate"]

METHOD_ORDER = [
    "base_clean", "base_aug", "plugin",
    "r3f", "r3f_plugin",
    "smart", "smart_plugin",
    "awp", "awp_plugin",
]

PERTURB_ORDER = ["clean", "typo", "distractor", "format_rewrite"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma_q", type=float, default=DEFAULT_GAMMA_Q)
    return p.parse_args()


def load_all_csvs(results_dir: str) -> list:
    rows = []
    for fp in glob.glob(os.path.join(results_dir, "results_*.csv")):
        if "combined" in fp or "gamma_sweep" in fp:
            continue
        with open(fp) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def group_rows(rows: list, gamma_q: float) -> dict:
    """Group rows by (method, perturbation); filter by gamma_q for plugin methods."""
    g = defaultdict(list)
    for r in rows:
        method = r["method"]
        ptype  = r["perturbation"]
        gq     = float(r.get("gamma_q", gamma_q))
        # For plugin methods, only keep the target gamma_q
        if method.endswith("_plugin") or method == "plugin":
            if abs(gq - gamma_q) > 1e-6:
                continue
        g[(method, ptype)].append(r)
    return g


def mean_std(vals: list):
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt(m, s, decimals=4):
    if m != m:  # nan
        return "N/A"
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


# ── Table 1: Main results ──────────────────────────────────────────────────

def make_table1(grouped: dict) -> list:
    """
    Rows: (method, perturbation)
    Cols: accuracy, worst_class_acc, worst_class_err, vwr_gamma, sigma_max, drop
    """
    header = ["method", "perturbation",
              "acc (mean±std)", "worst_acc (mean±std)",
              "worst_err (mean±std)", "VWR_gamma (mean±std)",
              "sigma_max (mean±std)", "drop (mean±std)"]
    table  = [header]

    for method in METHOD_ORDER:
        for ptype in PERTURB_ORDER:
            rows = grouped.get((method, ptype), [])
            if not rows:
                continue
            def col(key):
                vals = [float(r[key]) for r in rows
                        if r.get(key) not in ("", "nan", None)]
                return mean_std(vals)

            table.append([
                method, ptype,
                fmt(*col("accuracy")),
                fmt(*col("worst_class_acc")),
                fmt(*col("worst_class_err")),
                fmt(*col("vwr_gamma")),
                fmt(*col("sigma_max")),
                fmt(*col("clean_to_robust_drop")),
            ])
    return table


# ── Table 2: Plugin gain ───────────────────────────────────────────────────

_PLUGIN_PAIRS = [
    ("base_aug",  "plugin"),
    ("r3f",       "r3f_plugin"),
    ("smart",     "smart_plugin"),
    ("awp",       "awp_plugin"),
]


def make_table2(grouped: dict) -> list:
    """Show delta (plugin - base) for each pair."""
    header = ["base_method", "perturbation",
              "Δacc", "Δworst_acc", "Δworst_err",
              "ΔVWR_gamma", "Δsigma_max"]
    table  = [header]

    for base_m, plug_m in _PLUGIN_PAIRS:
        for ptype in PERTURB_ORDER:
            base_rows = grouped.get((base_m, ptype), [])
            plug_rows = grouped.get((plug_m, ptype), [])
            if not base_rows or not plug_rows:
                continue

            def delta(key):
                b = [float(r[key]) for r in base_rows
                     if r.get(key) not in ("", "nan", None)]
                p = [float(r[key]) for r in plug_rows
                     if r.get(key) not in ("", "nan", None)]
                if not b or not p:
                    return float("nan")
                return statistics.mean(p) - statistics.mean(b)

            table.append([
                base_m, ptype,
                f"{delta('accuracy'):+.4f}",
                f"{delta('worst_class_acc'):+.4f}",
                f"{delta('worst_class_err'):+.4f}",
                f"{delta('vwr_gamma'):+.4f}",
                f"{delta('sigma_max'):+.4f}",
            ])
    return table


# ── Table 3: Ablation (if data exists) ────────────────────────────────────

def make_table3(grouped: dict) -> list:
    ablation_methods = ["base_aug", "plugin_rspec_only",
                        "plugin_rstab_only", "plugin"]
    header = ["method", "perturbation",
              "acc", "worst_acc", "VWR_gamma", "sigma_max"]
    table  = [header]
    any_data = False
    for m in ablation_methods:
        for ptype in PERTURB_ORDER:
            rows = grouped.get((m, ptype), [])
            if not rows:
                continue
            any_data = True
            def col(key):
                vals = [float(r[key]) for r in rows
                        if r.get(key) not in ("", "nan", None)]
                return mean_std(vals)
            table.append([m, ptype,
                          fmt(*col("accuracy")),
                          fmt(*col("worst_class_acc")),
                          fmt(*col("vwr_gamma")),
                          fmt(*col("sigma_max"))])
    if not any_data:
        return []
    return table


# ── Gamma sweep plot data ──────────────────────────────────────────────────

def make_gamma_sweep_data(rows: list) -> dict:
    """
    Build per-method-perturbation curves over gamma_q values.
    Returns dict ready to serialise as JSON.
    """
    plugin_methods = ["plugin", "r3f_plugin", "smart_plugin", "awp_plugin"]
    plot = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    metrics_plot = ["accuracy", "worst_class_acc", "vwr_gamma", "sigma_max"]

    from collections import defaultdict as dd
    g = dd(list)
    for r in rows:
        if r["method"] in plugin_methods:
            g[(r["method"], r.get("gamma_q","?"), r["perturbation"])].append(r)

    for (method, gq, ptype), rlist in g.items():
        for m in metrics_plot:
            vals = [float(r[m]) for r in rlist if r.get(m) not in ("","nan",None)]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            plot[method][ptype][str(gq)][m] = {"mean": round(mean, 4),
                                                "std":  round(std,  4)}
    return dict(plot)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    gamma_q = args.gamma_q
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = load_all_csvs(RESULTS_DIR)
    if not rows:
        print("No result CSVs found in", RESULTS_DIR)
        return
    print(f"  Loaded {len(rows)} rows from {RESULTS_DIR}")

    # ── Combined CSV ───────────────────────────────────────────────────
    combined = os.path.join(RESULTS_DIR, "results_combined.csv")
    with open(combined, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → results_combined.csv  ({len(rows)} rows)")

    grouped = group_rows(rows, gamma_q)

    # ── Table 1 ────────────────────────────────────────────────────────
    t1 = make_table1(grouped)
    with open(os.path.join(RESULTS_DIR, "table1_main.csv"), "w", newline="") as f:
        csv.writer(f).writerows(t1)
    print(f"  → table1_main.csv  ({len(t1)-1} rows)")

    # ── Table 2 ────────────────────────────────────────────────────────
    t2 = make_table2(grouped)
    with open(os.path.join(RESULTS_DIR, "table2_plugin_gain.csv"), "w", newline="") as f:
        csv.writer(f).writerows(t2)
    print(f"  → table2_plugin_gain.csv  ({len(t2)-1} rows)")

    # ── Table 3 ────────────────────────────────────────────────────────
    t3 = make_table3(grouped)
    if t3:
        with open(os.path.join(RESULTS_DIR, "table3_ablation.csv"), "w", newline="") as f:
            csv.writer(f).writerows(t3)
        print(f"  → table3_ablation.csv  ({len(t3)-1} rows)")

    # ── Gamma sweep JSON ───────────────────────────────────────────────
    sweep = make_gamma_sweep_data(rows)
    with open(os.path.join(RESULTS_DIR, "gamma_sweep_plot_data.json"), "w") as f:
        json.dump(sweep, f, indent=2)
    print("  → gamma_sweep_plot_data.json")

    # ── Print Table 1 to stdout ────────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 1  — Main Results  (mean±std, gamma_q={})".format(gamma_q))
    print("─" * 120)
    col_widths = [18, 16, 16, 16, 16, 18, 18, 14]
    for row in t1:
        line = "  ".join(str(c).ljust(w)
                         for c, w in zip(row, col_widths))
        print(line)

    print("\n" + "─" * 100)
    print("TABLE 2  — Plugin Gain")
    print("─" * 100)
    col_widths2 = [14, 16, 10, 10, 10, 14, 14]
    for row in t2:
        line = "  ".join(str(c).ljust(w)
                         for c, w in zip(row, col_widths2))
        print(line)


if __name__ == "__main__":
    main()
