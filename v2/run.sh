#!/bin/bash
# ─────────────────────────────────────────────────────────────
# PAC-robust v2  —  Quick-start script
# ─────────────────────────────────────────────────────────────
#
# 1. Smoke test (no GPU)
# 2. Priority-1 experiments (2 seeds)
# 3. Gamma sweep (q10/q25/q50)
# 4. Aggregate results + print tables
#
# Usage:
#   cd /LOCAL2/zhuoyun/PAC_robust/v2
#   bash run.sh [--p2] [--device cuda:0]
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python
DEVICE=cuda:0
P2=0

# Parse args
for arg in "$@"; do
  case $arg in
    --p2)     P2=1 ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --device)   shift; DEVICE="$1" ;;
  esac
done

cd /LOCAL2/zhuoyun/PAC_robust/v2

echo "══════════════════════════════════════════════════════════"
echo "  PAC-robust v2  |  device=$DEVICE"
echo "══════════════════════════════════════════════════════════"

# ── Smoke test ──────────────────────────────────────────────
echo ""
echo "── Smoke test ──"
$PYTHON scripts/smoke_test.py

# ── Priority-1 full run ─────────────────────────────────────
echo ""
echo "── Priority-1 experiments ──"
P2_FLAG=""
[ $P2 -eq 1 ] && P2_FLAG="--p2"

$PYTHON scripts/run_all.py --device "$DEVICE" $P2_FLAG

# ── Gamma sweep ─────────────────────────────────────────────
echo ""
echo "── Gamma sweep ──"
$PYTHON scripts/gamma_sweep.py --device "$DEVICE"

# ── Aggregate ───────────────────────────────────────────────
echo ""
echo "── Aggregate results ──"
$PYTHON scripts/aggregate.py

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done.  Results in: /LOCAL2/zhuoyun/PAC_robust/v2/results/"
echo "══════════════════════════════════════════════════════════"
