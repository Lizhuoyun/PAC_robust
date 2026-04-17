#!/bin/bash
cd /LOCAL2/zhuoyun/PAC_robust/v2
PYTHON=/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python
pkill -9 -f alpha_sweep_7b 2>/dev/null || true
pkill -9 -f run_experiment 2>/dev/null || true
sleep 3
nohup $PYTHON -u scripts/alpha_sweep_7b.py > logs/alpha_sweep_7b_main_v2.log 2>&1 &
echo "PID=$!"
sleep 10
head -5 logs/alpha_sweep_7b_main_v2.log
