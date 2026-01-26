#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for GSM8K (generation) for a single model.
# Runs a small suite of robust plugins (augment/r3f/smart/spectral) and evaluates
# robust EM over eval_budgets (small/medium/large) + token_risk/sigma_max.
#
# Usage:
#   bash experiments/run_gsm8k_suite_multigpu.sh
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=gsm8k_qwen7b_suite_v1
#   WANDB_MODE=online   (or offline/disabled)
#   GPUS="0,1,2"
#   MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
#   RESULTS_ROOT=results/gsm8k/qwen7b_suite_v1
#   SEED=42
#   MODEL_DTYPE=bf16
#   MAX_LENGTH=512
#   BATCH_SIZE=1
#   EPOCHS=1
#   LORA_R=16
#   LORA_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj"]'
#
# Optional data caps (handy for debugging):
#   MAX_TRAIN_EXAMPLES=0 (0 means full)
#   MAX_EVAL_EXAMPLES=0  (0 means full)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
fi

export HF_HOME="${HF_HOME:-/LOCAL2/zhuoyun/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${NLTK_DATA:-/LOCAL2/zhuoyun/nltk_data}"

WANDB_PROJECT="${WANDB_PROJECT:-icml_wcr_spectral}"
WANDB_GROUP="${WANDB_GROUP:-gsm8k_suite_v1}"
WANDB_MODE="${WANDB_MODE:-online}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified (GPUS=${GPUS_RAW})"
  exit 1
fi

MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SEED="${SEED:-42}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1}"
LORA_R="${LORA_R:-16}"
LORA_TARGET="${LORA_TARGET_MODULES:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]}"

MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-0}"

CONFIG="${CONFIG_PATH:-configs/generation/gsm8k/nll.yaml}"

# Short model tag for W&B naming (avoid slashes/spaces)
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="$(echo "${MODEL_TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g')"

RESULTS_ROOT="${RESULTS_ROOT:-results/gsm8k/${MODEL_TAG}_suite_v1}"
LOG_ROOT="${RESULTS_ROOT}/_logs"
mkdir -p "${LOG_ROOT}"

# Each entry: name; augment_enabled; r3f_enabled; smart_enabled; spectral_enabled
RUNS=(
  "erm;false;false;false;false"
  "erm_augment;true;false;false;false"
  "erm_r3f;false;true;false;false"
  "erm_smart;false;false;true;false"
  "erm_spectral;false;false;false;true"
  "erm_smart_spectral;false;false;true;true"
)

run_one() {
  local gpu="$1"
  local run_name="$2"
  local augment="$3"
  local r3f="$4"
  local smart="$5"
  local spectral="$6"

  local out_dir="${RESULTS_ROOT}/${run_name}/seed${SEED}"
  local log_file="${LOG_ROOT}/${run_name}_seed${SEED}_gpu${gpu}.log"
  mkdir -p "${out_dir}"

  echo "[run] gpu=${gpu} model=${MODEL_TAG} run=${run_name} -> ${out_dir}"

  local overrides=()
  overrides+=(--override "seed=${SEED}")
  overrides+=(--override "model_name_or_path=${MODEL}")
  overrides+=(--override "model.torch_dtype=${MODEL_DTYPE}")
  overrides+=(--override "max_length=${MAX_LENGTH}")
  overrides+=(--override "batch_size=${BATCH_SIZE}")
  overrides+=(--override "epochs=${EPOCHS}")
  overrides+=(--override "finetune_mode=lora")
  overrides+=(--override "lora.target_modules=${LORA_TARGET}")
  overrides+=(--override "lora.r=${LORA_R}")
  overrides+=(--override "augment.enabled=${augment}")
  overrides+=(--override "augment.preset=standard")
  overrides+=(--override "r3f.enabled=${r3f}")
  overrides+=(--override "r3f.preset=standard")
  overrides+=(--override "smart.enabled=${smart}")
  overrides+=(--override "smart.preset=standard")
  overrides+=(--override "spectral.enabled=${spectral}")
  overrides+=(--override "spectral.alpha=0.05")
  overrides+=(--override "logging.backend=both")
  overrides+=(--override "logging.wandb.project=${WANDB_PROJECT}")
  overrides+=(--override "logging.wandb.group=${WANDB_GROUP}")
  overrides+=(--override "logging.wandb.mode=${WANDB_MODE}")
  overrides+=(--override "logging.save_dir=${out_dir}")

  if [[ "${MAX_TRAIN_EXAMPLES}" != "0" ]]; then
    overrides+=(--override "dataset.max_train_examples=${MAX_TRAIN_EXAMPLES}")
  fi
  if [[ "${MAX_EVAL_EXAMPLES}" != "0" ]]; then
    overrides+=(--override "dataset.max_eval_examples=${MAX_EVAL_EXAMPLES}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m experiments.train_generation \
    --config "${CONFIG}" \
    "${overrides[@]}" \
    --override "logging.wandb.name=${MODEL_TAG}/${run_name}/seed${SEED}/train" \
    --override "logging.metrics_path=${out_dir}/train_metrics.jsonl" \
    --override "logging.final_metrics_path=${out_dir}/train_metrics.json" \
    > "${log_file}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m experiments.eval_generation \
    --config "${CONFIG}" \
    --ckpt "${out_dir}" \
    "${overrides[@]}" \
    --override "logging.wandb.name=${MODEL_TAG}/${run_name}/seed${SEED}/eval" \
    --override "logging.metrics_path=${out_dir}/eval_metrics.jsonl" \
    --override "logging.final_metrics_path=${out_dir}/eval_metrics.json" \
    >> "${log_file}" 2>&1
}

# One-job-per-GPU scheduler (same pattern as ARC scripts).
declare -A PID_BY_GPU
job_idx=0

for spec in "${RUNS[@]}"; do
  IFS=';' read -r run_name augment r3f smart spectral <<< "${spec}"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    wait "${prev}" || true
    unset PID_BY_GPU["$gpu"]
  fi
  (run_one "${gpu}" "${run_name}" "${augment}" "${r3f}" "${smart}" "${spectral}") &
  PID_BY_GPU["$gpu"]=$!
  job_idx=$((job_idx + 1))
  sleep 2
done

echo "[wait] waiting for remaining runs to finish..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
echo "[done] all runs completed. results: ${RESULTS_ROOT}"


