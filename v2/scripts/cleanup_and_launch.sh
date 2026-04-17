#!/bin/bash
set -x

echo "=== Step 1: Find GPU processes ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo "=== Step 2: Kill scheduler ==="
ps aux | grep alpha_sweep_7b | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

echo "=== Step 3: Kill training processes ==="
ps aux | grep 'run_experiment.py' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

echo "=== Step 4: Wait for GPU memory to free ==="
sleep 10

echo "=== Step 5: Check GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo "=== Step 6: Launch scheduler ==="
cd /LOCAL2/zhuoyun/PAC_robust/v2
nohup /LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python -u scripts/alpha_sweep_7b.py > logs/alpha_sweep_7b_main_v3.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler PID: $SCHEDULER_PID"

echo "=== Step 7: Wait for initial job start ==="
sleep 30

echo "=== Step 8: Initial check ==="
tail -10 /LOCAL2/zhuoyun/PAC_robust/v2/logs/alpha_sweep_7b_main_v3.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "=== LAUNCH COMPLETE ==="
