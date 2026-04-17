#!/bin/bash
set -e

cd /LOCAL2/zhuoyun/PAC_robust/v2
PYTHON=/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python

echo "=== Killing existing processes ==="
pkill -9 -f "alpha_sweep_7b" 2>/dev/null || true
pkill -9 -f "run_experiment" 2>/dev/null || true
sleep 3

echo "=== GPU Info ==="
$PYTHON -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_mem/(1024**3):.1f} GB')
"

echo "=== Job Count ==="
$PYTHON -c "
import sys, os; sys.path.insert(0, '.')
from scripts.alpha_sweep_7b import build_jobs, MAX_PER_GPU
jobs = build_jobs()
skip = sum(1 for j in jobs if os.path.exists(j.result_csv))
print(f'MAX_PER_GPU={MAX_PER_GPU}')
print(f'Total={len(jobs)}, Skip={skip}, ToRun={len(jobs)-skip}')
"

echo "=== Launching Scheduler ==="
nohup $PYTHON -u scripts/alpha_sweep_7b.py > logs/alpha_sweep_7b_main.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler PID: $SCHEDULER_PID"

sleep 30
echo "=== Scheduler Log ==="
tail -20 logs/alpha_sweep_7b_main.log

echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "=== Done ==="
