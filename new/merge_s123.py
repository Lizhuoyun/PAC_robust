#!/usr/bin/env python3
import csv, sys

new_rows = []
with open('results/results_robot_s123.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        for k in ['accuracy','worst_class_acc','worst_class_err','vwr_gamma','sigma_max','gamma','alpha','beta','kappa']:
            v = float(r[k])
            r[k] = f'{v:.3f}'
        r['seed'] = str(int(float(r['seed'])))
        new_rows.append(r)

existing = []
with open('results/results_all.csv') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    existing = list(reader)

existing.extend(new_rows)

with open('results/results_all.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(existing)

print(f"Merged: {len(existing)} total rows ({len(new_rows)} new robot s123)")
