#!/bin/bash
# Master run script for PAC-Robust Plugin Regulariser feasibility study.
#
# Usage:
#   bash run.sh                  # Run all tasks (agnews, arc, robot) with 2 seeds
#   bash run.sh --tasks agnews   # Run only AG News
#   bash run.sh --device cuda:1  # Use a different GPU
#
# Prerequisites:
#   The venv at /LOCAL2/zhuoyun/Robustness_fairness/venv must have:
#   torch, transformers, peft, bitsandbytes, datasets, pillow, matplotlib, seaborn, pandas

PYTHON=/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HF_HOME=/LOCAL2/zhuoyun/hf_cache
export TRANSFORMERS_CACHE=/LOCAL2/zhuoyun/hf_cache/hub
export TOKENIZERS_PARALLELISM=false

cd "$SCRIPT_DIR"

echo "============================================================"
echo "PAC-Robust Plugin Regulariser — Feasibility Study"
echo "============================================================"
echo "Python: $PYTHON"
echo "Working dir: $SCRIPT_DIR"
echo "Start time: $(date)"
echo ""

$PYTHON run_all.py "$@"

echo ""
echo "Finished at: $(date)"
echo "Results directory: $SCRIPT_DIR/results/"
