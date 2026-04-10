#!/usr/bin/env bash
set -euo pipefail

# ARC full suite (8 methods x 3 budgets) for 3 large models, queued per GPU.
# Requirement: adversarial+spectral combos use INNER ONLY:
#   - erm_r3f_spectral  -> r3f.spectral_guided=true, spectral.inner_guided=true, spectral.enabled=false
#   - erm_smart_spectral -> smart.spectral_guided=true, spectral.inner_guided=true, spectral.enabled=false
#
# Models:
#   Qwen/Qwen2.5-7B-Instruct
#   meta-llama/Llama-3.1-8B-Instruct
#   mistralai/Mistral-7B-Instruct-v0.3
#
# Env:
#   VENV=/LOCAL2/zhuoyun/Robustness_fairness/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   GPUS="0,1,2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustness_fairness/venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] cannot find python in VENV: ${PYTHON}"
  exit 2
fi

export HF_HOME="${HF_HOME:-/LOCAL2/zhuoyun/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${NLTK_DATA:-/LOCAL2/zhuoyun/nltk_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

BUDGETS=("small" "medium" "large")

# 8 methods: ERM, ERM+spectral, ERM+R3F, ERM+R3F+spectral, ERM+SMART, ERM+SMART+spectral, Augment, Augment+spectral
RUNS=(
  "erm;false;false;false;false"
  "erm_spectral;false;false;true;false"
  "erm_r3f;true;false;false;false"
  "erm_r3f_spectral;true;false;true;false"
  "erm_smart;false;true;false;false"
  "erm_smart_spectral;false;true;true;false"
  "erm_augment;false;false;false;true"
  "erm_augment_spectral;false;false;true;true"
)

tagify() {
  local m="$1"
  local tag="${m##*/}"
  echo "${tag}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g'
}

is_done_json() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  grep -q "\"status\"\\s*:\\s*\"done\"" "$p" && return 0
  return 1
}

run_one() {
  local gpu="$1"
  local model="$2"
  local run_name="$3"
  local r3f="$4"
  local smart="$5"
  local spectral="$6"
  local augment="$7"
  local budget="$8"

  local model_tag
  model_tag="$(tagify "${model}")"
  local root="results/arc/${model_tag}_fullsuite_inneronly"
  local out="${root}/${run_name}/budget_${budget}/seed42"
  local log="${root}/_logs/${run_name}_budget_${budget}.log"
  mkdir -p "${root}/_logs" "${out}"

  if is_done_json "${out}/metrics.json"; then
    echo "[skip] ${out}"
    return 0
  fi

  # Inner-only switches for adversarial+spectral.
  local spectral_enabled="${spectral}"
  local spectral_inner_guided="false"
  local r3f_spectral_guided="false"
  local smart_spectral_guided="false"
  if [[ "${run_name}" == *"r3f_spectral"* ]]; then
    spectral_enabled="false"
    spectral_inner_guided="true"
    r3f_spectral_guided="true"
  elif [[ "${run_name}" == *"smart_spectral"* ]]; then
    spectral_enabled="false"
    spectral_inner_guided="true"
    smart_spectral_guided="true"
  fi

  echo "[run] gpu=${gpu} model=${model} run=${run_name} budget=${budget} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m experiments.train_classification \
    --config configs/classification/arc/erm.yaml \
    --log_backend jsonl \
    --override "seed=42" \
    --override "model_name_or_path=${model}" \
    --override "batch_size=1" \
    --override "max_length=256" \
    --override "epochs=1" \
    --override "model.torch_dtype=bf16" \
    --override "finetune_mode=lora" \
    --override "lora.target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]" \
    --override "augment.enabled=${augment}" \
    --override "augment.clean_ratio=0.5" \
    --override "augment.train_budget=${budget}" \
    --override "r3f.enabled=${r3f}" \
    --override "r3f.spectral_guided=${r3f_spectral_guided}" \
    --override "smart.enabled=${smart}" \
    --override "smart.spectral_guided=${smart_spectral_guided}" \
    --override "spectral.enabled=${spectral_enabled}" \
    --override "spectral.inner_guided=${spectral_inner_guided}" \
    --override "perturbation.budget=${budget}" \
    --override "logging.save_dir=${out}" \
    --override "logging.metrics_path=${out}/metrics.jsonl" \
    --override "logging.final_metrics_path=${out}/metrics.json" \
    --override "logging.matrix_path=${out}/matrix.json" \
    > "${log}" 2>&1
}

# Build job list: 3 models * 8 methods * 3 budgets = 72 jobs
JOBS=()
for m in "${MODELS[@]}"; do
  for spec in "${RUNS[@]}"; do
    IFS=';' read -r run_name r3f smart spectral augment <<< "${spec}"
    for b in "${BUDGETS[@]}"; do
      JOBS+=("${m}|${run_name}|${r3f}|${smart}|${spectral}|${augment}|${b}")
    done
  done
done

declare -A PID_BY_GPU
declare -A SPEC_BY_PID
FAILURES=0
job_idx=0

for spec in "${JOBS[@]}"; do
  IFS='|' read -r model run_name r3f smart spectral augment budget <<< "${spec}"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"

  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    if ! wait "${prev}"; then
      echo "[fail] job failed on gpu=${gpu}: ${SPEC_BY_PID[$prev]:-unknown}"
      FAILURES=$((FAILURES + 1))
    fi
    unset PID_BY_GPU["$gpu"]
  fi

  (run_one "${gpu}" "${model}" "${run_name}" "${r3f}" "${smart}" "${spectral}" "${augment}" "${budget}") &
  pid=$!
  PID_BY_GPU["$gpu"]=$pid
  SPEC_BY_PID["$pid"]="${spec}"
  job_idx=$((job_idx + 1))
  sleep 2
done

echo "[wait] waiting for remaining jobs..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    if ! wait "${pid}"; then
      echo "[fail] job failed on gpu=${gpu}: ${SPEC_BY_PID[$pid]:-unknown}"
      FAILURES=$((FAILURES + 1))
    fi
  fi
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "[done-with-failures] finished with ${FAILURES} failed job(s)."
  exit 1
fi
echo "[done] all ARC jobs completed successfully."

