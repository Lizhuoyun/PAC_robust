"""Merge task-specific result CSVs into one combined file."""
import os, sys, glob, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

def merge():
    combined = []
    header = None
    
    # Read all CSV files (results_*.csv and results.csv)
    for pattern in ["results_*.csv", "results.csv"]:
        for fpath in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            with open(fpath) as f:
                reader = csv.DictReader(f)
                if header is None:
                    header = reader.fieldnames
                for row in reader:
                    key = (row["task"], row["method"], row["seed"], row["perturbation"])
                    combined.append((key, row))
    
    # Deduplicate (keep last occurrence)
    seen = {}
    for key, row in combined:
        seen[key] = row
    
    rows = list(seen.values())
    if not rows:
        print("No results found!")
        return
    
    outpath = os.path.join(RESULTS_DIR, "results_combined.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    
    print(f"Merged {len(rows)} rows → {outpath}")
    
    # Summary
    from collections import Counter
    tasks = Counter(r["task"] for r in rows)
    for task, count in sorted(tasks.items()):
        print(f"  {task}: {count} rows")

if __name__ == "__main__":
    merge()
