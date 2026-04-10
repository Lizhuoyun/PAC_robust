"""
Combine all available results and generate analysis.
Run this after experiments complete.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR

def combine_results():
    """Read all CSV files and combine into one."""
    all_rows = []
    header = None
    seen_keys = set()

    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath) as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                key = (row["task"], row["method"], row["seed"], row["perturbation"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_rows.append(row)

    outpath = os.path.join(RESULTS_DIR, "results_all.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Combined {len(all_rows)} rows from {len(seen_keys)} unique experiments")
    return outpath


if __name__ == "__main__":
    csv_path = combine_results()
    
    from analyze import generate_all, load_results
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} results")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Seeds: {df['seed'].unique().tolist()}")
    
    generate_all(csv_path)
