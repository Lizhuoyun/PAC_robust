#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for Qwen/Qwen2.5-0.5B-Instruct on ARC (classification).
# Runs: ERM, ERM+spectral, ERM+r3f, ERM+r3f+spectral, ERM+smart, ERM+smart+spectral
# for multiple perturbation budgets (small/medium/large).
#
# Usage:
#   bash experiments/run_qwen05b_arc_full_multigpu.sh
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=qwen05b_arc_full
#   WANDB_MODE=online   (or offline/disabled)
#   GPUS="0,1,2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] cannot find python in VENV: PYTHON=${PYTHON} (set VENV or PYTHON)"
  exit 2
fi

export HF_HOME="${HF_HOME:-/LOCAL2/zhuoyun/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${NLTK_DATA:-/LOCAL2/zhuoyun/nltk_data}"

WANDB_PROJECT="${WANDB_PROJECT:-icml_wcr_spectral}"
WANDB_GROUP="${WANDB_GROUP:-qwen05b_arc_full}"
WANDB_MODE="${WANDB_MODE:-offline}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified (GPUS=${GPUS_RAW})"
  exit 1
fi

MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
LORA_TARGET="${LORA_TARGET_MODULES:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EPOCHS="${EPOCHS:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
INCLUDE_AUGMENT="${INCLUDE_AUGMENT:-0}"
AUGMENT_CLEAN_RATIO="${AUGMENT_CLEAN_RATIO:-0.5}"
RUN_SET="${RUN_SET:-all}"   # all | augment_only | spectral_only

# Short model tag for W&B naming (avoid slashes/spaces)
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="$(echo "${MODEL_TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g')"

TASK_CONFIG="${TASK_CONFIG:-configs/classification/arc/erm.yaml}"
DATASET_NAME="${DATASET_NAME:-arc}"

# Budgets control synonym replace ratio via perturbation.budget (small/medium/large).
BUDGETS=("small" "medium" "large")

# Each entry: name; r3f_enabled; smart_enabled; spectral_enabled; augment_enabled
RUNS=()
if [[ "${RUN_SET}" == "augment_only" ]]; then
  RUNS+=(
    "erm_augment;false;false;false;true"
    "erm_augment_spectral;false;false;true;true"
  )
elif [[ "${RUN_SET}" == "spectral_only" ]]; then
  RUNS+=(
    "erm_spectral;false;false;true;false"
    "erm_r3f_spectral;true;false;true;false"
    "erm_smart_spectral;false;true;true;false"
  )
  if [[ "${INCLUDE_AUGMENT}" == "1" ]]; then
    RUNS+=(
      "erm_augment_spectral;false;false;true;true"
    )
  fi
else
  RUNS+=(
    "erm;false;false;false;false"
    "erm_spectral;false;false;true;false"
    "erm_r3f;true;false;false;false"
    "erm_r3f_spectral;true;false;true;false"
    "erm_smart;false;true;false;false"
    "erm_smart_spectral;false;true;true;false"
  )
  if [[ "${INCLUDE_AUGMENT}" == "1" ]]; then
    RUNS+=(
      "erm_augment;false;false;false;true"
      "erm_augment_spectral;false;false;true;true"
    )
  fi
fi

RESULTS_ROOT="${RESULTS_ROOT:-results/${DATASET_NAME}/qwen05b_full}"
LOG_ROOT="${RESULTS_ROOT}/_logs"
mkdir -p "${LOG_ROOT}"

run_one() {
  local gpu="$1"
  local run_name="$2"
  local r3f="$3"
  local smart="$4"
  local spectral="$5"
  local augment="$6"
  local budget="$7"

  local out_dir="${RESULTS_ROOT}/${run_name}/budget_${budget}/seed${SEED}"
  local log_file="${LOG_ROOT}/${run_name}_budget_${budget}_seed${SEED}_gpu${gpu}.log"
  mkdir -p "${out_dir}"

  echo "[run] gpu=${gpu} run=${run_name} budget=${budget} -> ${out_dir}"

  # For adversarial+spectral combos, use inner-guided only (no external spectral).
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

  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PYTHON}" -m experiments.train_classification \
    --config "${TASK_CONFIG}" \
    --log_backend jsonl \
    --override "seed=${SEED}" \
    --override "model_name_or_path=${MODEL}" \
    --override "batch_size=${BATCH_SIZE}" \
    --override "max_length=${MAX_LENGTH}" \
    --override "epochs=${EPOCHS}" \
    --override "model.torch_dtype=${MODEL_DTYPE}" \
    --override "finetune_mode=lora" \
    --override "lora.target_modules=${LORA_TARGET}" \
    --override "augment.enabled=${augment}" \
    --override "augment.clean_ratio=${AUGMENT_CLEAN_RATIO}" \
    --override "augment.train_budget=${budget}" \
    --override "r3f.enabled=${r3f}" \
    --override "r3f.spectral_guided=${r3f_spectral_guided}" \
    --override "smart.enabled=${smart}" \
    --override "smart.spectral_guided=${smart_spectral_guided}" \
    --override "spectral.enabled=${spectral_enabled}" \
    --override "spectral.inner_guided=${spectral_inner_guided}" \
    --override "perturbation.budget=${budget}" \
    --override "perturbation.mix={\"typo\":1.0,\"synonym\":1.0,\"paraphrase\":1.0}" \
    --override "logging.save_dir=${out_dir}" \
    --override "logging.metrics_path=${out_dir}/metrics.jsonl" \
    --override "logging.final_metrics_path=${out_dir}/metrics.json" \
    --override "logging.matrix_path=${out_dir}/matrix.json" \
    > "${log_file}" 2>&1
}

# One-job-per-GPU scheduler:
# - Assign jobs round-robin to GPUs
# - Before launching a new job on a GPU, wait for the previous job on that GPU to finish
declare -A PID_BY_GPU
job_idx=0

for spec in "${RUNS[@]}"; do
  IFS=';' read -r run_name r3f smart spectral augment <<< "${spec}"
  for budget in "${BUDGETS[@]}"; do
    gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
    prev="${PID_BY_GPU[$gpu]:-}"
    if [[ -n "${prev}" ]]; then
      wait "${prev}" || true
      unset PID_BY_GPU["$gpu"]
    fi
    (run_one "${gpu}" "${run_name}" "${r3f}" "${smart}" "${spectral}" "${augment}" "${budget}") &
    PID_BY_GPU["$gpu"]=$!
    job_idx=$((job_idx + 1))
    # Small stagger to reduce simultaneous HF cache contention
    sleep 2
  done
done

echo "[wait] waiting for remaining runs to finish..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
echo "[done] all runs completed."


