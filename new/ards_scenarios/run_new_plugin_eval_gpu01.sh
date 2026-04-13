#!/usr/bin/env bash
# Evaluate new Plugin-LoRA checkpoints on ScienceQA test set.
# Uses only physical GPU 0 and 1 (CUDA_VISIBLE_DEVICES); GPU 2 left free.

set -euo pipefail

export PYTHONPATH="/LOCAL2/zhuoyun/PAC_robust/ARDS:/LOCAL2/zhuoyun/PAC_robust/new:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/LOCAL2/zhuoyun/hf_cache}"

ROOT="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"
CKPT="${ROOT}/checkpoints"
RES="${ROOT}/results"
PY="/LOCAL2/zhuoyun/PAC_robust/ards_venv/bin/python"
EVAL_PY="${ROOT}/sqa_unified_eval.py"

G10="-0.5371976017951965"
G25="-0.42339251935482025"
G50="-0.25576721131801605"

mkdir -p "$RES"

run_one () {
  local gpu="$1"   # 0 or 1 — passed to CUDA_VISIBLE_DEVICES
  local name="$2"  # q10 | q25 | q50
  local gamm="$3"
  local out_json="${RES}/eval_plugin_${name}.json"
  local logf="${RES}/eval_plugin_${name}_run.log"
  echo "=== ${name} on physical GPU ${gpu} -> ${out_json} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EVAL_PY}" \
    --ckpt_dir "${CKPT}/sqa_plugin_${name}" \
    --gamma "${gamm}" \
    --gpu 0 \
    --output "${out_json}" \
    2>&1 | tee "${logf}"
}

# Parallel: q10 + q25
run_one 0 q10 "${G10}" &
PID10=$!
run_one 1 q25 "${G25}" &
PID25=$!
wait "${PID10}" "${PID25}"

echo "=== q50 on physical GPU 0 ==="
run_one 0 q50 "${G50}"

echo "=== aggregate ==="
"${PY}" "${ROOT}/sqa_aggregate_results.py"

echo "=== done ==="
