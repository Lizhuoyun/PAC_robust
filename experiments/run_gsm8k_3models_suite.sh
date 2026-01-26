#!/usr/bin/env bash
set -euo pipefail

# Run GSM8K full suite (generation) for 3 models sequentially.
# Each model run uses all GPUs listed in GPUS for parallelizing the suite across runs.
#
# Usage:
#   bash experiments/run_gsm8k_3models_suite.sh
#
# Optional env:
#   GPUS="0,1,2"
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_MODE=online
#   SUITE_TAG=gsm8k_suite_v1
#   RESULTS_BASE=results/gsm8k

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUITE_TAG="${SUITE_TAG:-gsm8k_suite_v1}"
RESULTS_BASE="${RESULTS_BASE:-results/gsm8k}"

WANDB_PROJECT="${WANDB_PROJECT:-icml_wcr_spectral}"
WANDB_MODE="${WANDB_MODE:-online}"

GPUS="${GPUS:-0,1,2}"

run_model() {
  local model_name="$1"
  local model_tag="$2"
  local lora_r="$3"
  local batch_size="$4"

  echo "[suite] model=${model_name} tag=${model_tag}"
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${SUITE_TAG}_${model_tag}" \
  GPUS="${GPUS}" \
  MODEL_NAME_OR_PATH="${model_name}" \
  RESULTS_ROOT="${RESULTS_BASE}/${model_tag}_${SUITE_TAG}" \
  LORA_R="${lora_r}" \
  BATCH_SIZE="${batch_size}" \
  bash experiments/run_gsm8k_suite_multigpu.sh
}

# You can tweak batch_size/lora_r per model if needed.
run_model "Qwen/Qwen2.5-7B-Instruct" "qwen25_7b_instruct" "16" "1"
run_model "meta-llama/Llama-3.1-8B-Instruct" "llama31_8b_instruct" "16" "1"
run_model "mistralai/Mistral-7B-Instruct-v0.3" "mistral7b_instruct_v03" "16" "1"

echo "[suite][done] 3-model GSM8K suite finished."


