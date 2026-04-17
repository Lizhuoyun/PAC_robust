#!/bin/bash
# Auto-launch alpha sweep when GPU 0 and 1 become free
# Usage: nohup bash scripts/auto_launch_when_free.sh > logs/auto_launch.log 2>&1 &

cd /LOCAL2/zhuoyun/PAC_robust/v2
PYTHON=/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python
THRESHOLD=5000  # Consider GPU "free" if < 5000 MiB used

echo "[$(date)] Starting auto-launcher. Monitoring GPU 0 and 1..."
echo "Threshold: ${THRESHOLD} MiB"

while true; do
    MEM0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    MEM1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "[$(date)] GPU 0: ${MEM0} MiB, GPU 1: ${MEM1} MiB"
    
    if [ "$MEM0" -lt "$THRESHOLD" ] && [ "$MEM1" -lt "$THRESHOLD" ]; then
        echo "[$(date)] GPUs are free! Launching scheduler..."
        $PYTHON -u scripts/alpha_sweep_7b.py > logs/alpha_sweep_7b_main_v3.log 2>&1
        EXIT=$?
        echo "[$(date)] Scheduler finished with exit code $EXIT"
        echo "[$(date)] Checking results..."
        $PYTHON -c "
import os, glob, sys; sys.path.insert(0, '.')
from scripts.alpha_sweep_7b import build_jobs
jobs = build_jobs()
done = sum(1 for j in jobs if os.path.exists(j.result_csv))
print(f'Results: {done}/{len(jobs)}')
fail = [j.tag for j in jobs if not os.path.exists(j.result_csv)]
if fail:
    print(f'Missing: {len(fail)}')
    for t in fail[:10]:
        print(f'  {t}')
"
        break
    fi
    
    sleep 120  # Check every 2 minutes
done

echo "[$(date)] Auto-launcher complete."
