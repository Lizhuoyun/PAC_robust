#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], k: str) -> Any:
    return d.get(k, None)


def _parse_info(metrics_path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    Expected layout:
      results/arc/<suite>/<method>/<budget>/seed42/eval/metrics.json
    Returns: (suite, method, budget, seed)
    """
    parts = metrics_path.parts
    try:
        # find ".../results/arc/<suite>/..."
        arc_idx = None
        for i, p in enumerate(parts):
            if p == "arc" and i > 0 and parts[i - 1] == "results":
                arc_idx = i
                break
        if arc_idx is None:
            return None
        suite = parts[arc_idx + 1]
        # method/budget/seed by relative positions
        # .../<suite>/<method>/<budget>/seed42/eval/metrics.json
        method = parts[arc_idx + 2]
        budget = parts[arc_idx + 3]
        seed = parts[arc_idx + 4]
        if not budget.startswith("budget_"):
            return None
        if not seed.startswith("seed"):
            return None
        return suite, method, budget, seed
    except Exception:
        return None


def _find_metrics(roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        if r.is_file() and r.name == "metrics.json" and r.parent.name == "eval":
            out.append(r)
            continue
        for p in r.rglob("eval/metrics.json"):
            out.append(p)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        default=["results/arc"],
        help="Directories to search for eval/metrics.json.",
    )
    ap.add_argument(
        "--out",
        default="results/arc/_summary_arc_eval.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    roots = [Path(x) for x in args.roots]
    paths = _find_metrics(roots)
    if not paths:
        print("[summarize] no eval/metrics.json found under roots:", [str(r) for r in roots])
        return

    rows: List[Dict[str, Any]] = []
    for mp in paths:
        info = _parse_info(mp)
        if info is None:
            continue
        suite, method, budget, seed = info
        try:
            m = _read_json(mp)
        except Exception as e:
            print(f"[warn] failed to read {mp}: {e}")
            continue

        row: Dict[str, Any] = {
            "suite": suite,
            "method": method,
            "budget": budget,
            "seed": seed,
            "clean_acc": _safe_get(m, "clean_acc"),
            "robust_acc_small": _safe_get(m, "robust_acc_small"),
            "robust_acc_medium": _safe_get(m, "robust_acc_medium"),
            "robust_acc_large": _safe_get(m, "robust_acc_large"),
            "wcr_small": _safe_get(m, "wcr_small"),
            "wcr_medium": _safe_get(m, "wcr_medium"),
            "wcr_large": _safe_get(m, "wcr_large"),
            "wcr_plain_small": _safe_get(m, "wcr_plain_small"),
            "wcr_plain_medium": _safe_get(m, "wcr_plain_medium"),
            "wcr_plain_large": _safe_get(m, "wcr_plain_large"),
            "sigma_max_small": _safe_get(m, "sigma_max_small"),
            "sigma_max_medium": _safe_get(m, "sigma_max_medium"),
            "sigma_max_large": _safe_get(m, "sigma_max_large"),
            "path": str(mp),
        }
        rows.append(row)

    rows.sort(key=lambda r: (str(r["suite"]), str(r["method"]), str(r["budget"]), str(r["seed"])))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[summarize] wrote {len(rows)} rows -> {out_path}")
    # Print a small preview grouped by suite
    shown = 0
    last_suite = None
    for r in rows:
        if r["suite"] != last_suite:
            print(f"\n=== {r['suite']} ===")
            last_suite = r["suite"]
        print(
            f"- {r['method']:18s} | {r['budget']:12s} | {r['seed']:6s} | "
            f"clean {float(r['clean_acc'] or 0):.4f} | "
            f"rob(s/m/l) {float(r['robust_acc_small'] or 0):.4f}/{float(r['robust_acc_medium'] or 0):.4f}/{float(r['robust_acc_large'] or 0):.4f} | "
            f"wcr_plain(l) {float(r['wcr_plain_large'] or 0):.4f}"
        )
        shown += 1
        if shown >= 25:
            break
    if len(rows) > shown:
        print(f"... ({len(rows) - shown} more)")


if __name__ == "__main__":
    main()

