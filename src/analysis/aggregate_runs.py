import glob
import json
import os
from typing import Dict, List

import numpy as np

from src.train.utils import save_json


def _load_metrics(paths: List[str]) -> List[Dict]:
    out = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _aggregate(metrics: List[Dict]) -> Dict:
    keys = sorted({k for m in metrics for k in m.keys() if isinstance(m.get(k), (int, float))})
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics if k in m]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f}+/-{std:.4f}"


def _write_csv(path: str, summary: Dict) -> None:
    keys = sorted(summary.keys())
    metrics = sorted({m for k in keys for m in summary[k].keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write("run," + ",".join(metrics) + "\n")
        for run in keys:
            row = [run]
            for m in metrics:
                if m in summary[run]:
                    row.append(_format_mean_std(summary[run][m]["mean"], summary[run][m]["std"]))
                else:
                    row.append("")
            f.write(",".join(row) + "\n")


def _write_latex(path: str, summary: Dict) -> None:
    keys = sorted(summary.keys())
    metrics = sorted({m for k in keys for m in summary[k].keys()})
    with open(path, "w", encoding="utf-8") as f:
        cols = "l" + "c" * len(metrics)
        f.write("\\begin{tabular}{" + cols + "}\\n")
        f.write("Run & " + " & ".join(metrics) + " \\\\\\n")
        f.write("\\hline\\n")
        for run in keys:
            vals = []
            for m in metrics:
                if m in summary[run]:
                    vals.append(_format_mean_std(summary[run][m]["mean"], summary[run][m]["std"]))
                else:
                    vals.append("-")
            f.write(run + " & " + " & ".join(vals) + " \\\\\\n")
        f.write("\\end{tabular}\\n")


def aggregate_runs(input_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics_paths = glob.glob(os.path.join(input_dir, "**", "metrics.json"), recursive=True)
    grouped = {}
    for path in metrics_paths:
        key = os.path.basename(os.path.dirname(path))
        grouped.setdefault(key, []).append(path)

    summary = {}
    for key, paths in grouped.items():
        metrics = _load_metrics(paths)
        summary[key] = _aggregate(metrics)
        save_json(os.path.join(out_dir, f"{key}.json"), summary[key])

    save_json(os.path.join(out_dir, "summary.json"), summary)
    _write_csv(os.path.join(out_dir, "summary.csv"), summary)
    _write_latex(os.path.join(out_dir, "summary.tex"), summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    aggregate_runs(args.input, args.out)
