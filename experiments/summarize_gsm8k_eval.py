#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_get(d: Dict[str, Any], k: str) -> Any:
    return d.get(k, None)


def _parse_run_info(metrics_path: Path) -> Tuple[str, str, str]:
    """
    Expect layout like:
      results/gsm8k/<model_tag>_<suite_tag>/<run_name>/seed42/eval_metrics.json
    Returns: (model_root, run_name, seed_tag)
    """
    seed_tag = metrics_path.parent.name  # seed42
    run_name = metrics_path.parent.parent.name
    model_root = metrics_path.parent.parent.parent.name
    return model_root, run_name, seed_tag


def _find_metrics(roots: List[Path], metrics_name: str) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        if r.is_file() and r.name == metrics_name:
            out.append(r)
            continue
        for p in r.rglob(metrics_name):
            out.append(p)
    return sorted(set(out))


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        default=["results/gsm8k"],
        help="Directories (or files) to search for metrics json files.",
    )
    ap.add_argument(
        "--out",
        default="results/gsm8k/_summary_gsm8k_suite_v1.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--metrics_name",
        default="eval_metrics.json",
        help="Metrics filename to collect (e.g., eval_metrics.json or eval_metrics_v2.json).",
    )
    args = ap.parse_args()

    roots = [Path(x) for x in args.roots]
    paths = _find_metrics(roots, metrics_name=args.metrics_name)
    if not paths:
        print(f"[summarize] no {args.metrics_name} found under roots:", [str(r) for r in roots])
        return

    rows: List[Dict[str, Any]] = []
    for mp in paths:
        try:
            m = _read_json(mp)
        except Exception as e:
            print(f"[warn] failed to read {mp}: {e}")
            continue
        model_root, run_name, seed_tag = _parse_run_info(mp)
        row: Dict[str, Any] = {
            "model_root": model_root,
            "run": run_name,
            "seed": seed_tag,
            "clean_em": _safe_get(m, "clean_em"),
            "robust_em_small": _safe_get(m, "robust_em_small"),
            "robust_em_medium": _safe_get(m, "robust_em_medium"),
            "robust_em_large": _safe_get(m, "robust_em_large"),
            "token_risk_small": _safe_get(m, "token_risk_small"),
            "token_risk_medium": _safe_get(m, "token_risk_medium"),
            "token_risk_large": _safe_get(m, "token_risk_large"),
            "sigma_max_small": _safe_get(m, "sigma_max_small"),
            "sigma_max_medium": _safe_get(m, "sigma_max_medium"),
            "sigma_max_large": _safe_get(m, "sigma_max_large"),
            "path": str(mp),
        }
        rows.append(row)

    # Sort for readability
    rows.sort(key=lambda r: (str(r.get("model_root")), str(r.get("run")), str(r.get("seed"))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print a compact view to stdout
    print(f"[summarize] wrote {len(rows)} rows -> {out_path}")
    head = min(20, len(rows))
    for r in rows[:head]:
        print(
            f"- {r['model_root']:35s} | {r['run']:18s} | {r['seed']:6s} | "
            f"clean {r['clean_em']:.4f} | rob(s/m/l) {r['robust_em_small']:.4f}/{r['robust_em_medium']:.4f}/{r['robust_em_large']:.4f}"
        )
    if len(rows) > head:
        print(f"... ({len(rows) - head} more)")


if __name__ == "__main__":
    main()

