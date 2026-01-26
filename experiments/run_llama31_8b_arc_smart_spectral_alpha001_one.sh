#!/usr/bin/env bash
set -euo pipefail

# Single-run launcher:
#   ARC (classification) + Llama-3.1-8B-Instruct
#   method: ERM + SMART + Spectral
#   budget: large
#   spectral.alpha = 0.01
#
# Outputs (no overwrite):
#   results/arc/llama31_8b_smart_spectral_alpha001/erm_smart_spectral/budget_large/seed42/*
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=llama31_arc_alpha001
#   WANDB_MODE=online  (or offline/disabled)
#   GPU=0

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
WANDB_GROUP="${WANDB_GROUP:-llama31_arc_alpha001}"
WANDB_MODE="${WANDB_MODE:-online}"

GPU="${GPU:-0}"

MODEL="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
LORA_TARGET="${LORA_TARGET_MODULES:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EPOCHS="${EPOCHS:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

TASK_CONFIG="${TASK_CONFIG:-configs/classification/arc/erm.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/arc/llama31_8b_smart_spectral_alpha001}"
RUN_NAME="erm_smart_spectral"
BUDGET="large"

# Short model tag for W&B naming (avoid slashes/spaces)
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="$(echo "${MODEL_TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g')"

OUT_DIR="${RESULTS_ROOT}/${RUN_NAME}/budget_${BUDGET}/seed${SEED}"
LOG_ROOT="${RESULTS_ROOT}/_logs"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
TRAIN_LOG="${LOG_ROOT}/${RUN_NAME}_budget_${BUDGET}_seed${SEED}_train_gpu${GPU}.log"
EVAL_LOG="${LOG_ROOT}/${RUN_NAME}_budget_${BUDGET}_seed${SEED}_eval_gpu${GPU}.log"

echo "[train] gpu=${GPU} -> ${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" \
python -m experiments.train_classification \
  --config "${TASK_CONFIG}" \
  --log_backend wandb \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_name "${MODEL_TAG}/${RUN_NAME}/budget_${BUDGET}/seed${SEED}" \
  --override "seed=${SEED}" \
  --override "model_name_or_path=${MODEL}" \
  --override "batch_size=${BATCH_SIZE}" \
  --override "max_length=${MAX_LENGTH}" \
  --override "epochs=${EPOCHS}" \
  --override "model.torch_dtype=${MODEL_DTYPE}" \
  --override "finetune_mode=lora" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "augment.enabled=false" \
  --override "r3f.enabled=false" \
  --override "smart.enabled=true" \
  --override "spectral.enabled=true" \
  --override "spectral.alpha=0.01" \
  --override "perturbation.budget=${BUDGET}" \
  --override "perturbation.mix={\"typo\":1.0,\"synonym\":1.0,\"paraphrase\":1.0}" \
  --override "logging.save_dir=${OUT_DIR}" \
  --override "logging.metrics_path=${OUT_DIR}/metrics.jsonl" \
  --override "logging.final_metrics_path=${OUT_DIR}/metrics.json" \
  --override "logging.matrix_path=${OUT_DIR}/matrix.json" \
  > "${TRAIN_LOG}" 2>&1

echo "[eval] gpu=${GPU} ckpt=${OUT_DIR}"
mkdir -p "${OUT_DIR}/eval"
CUDA_VISIBLE_DEVICES="${GPU}" \
python -m experiments.eval_classification \
  --config "${OUT_DIR}/config_resolved.yaml" \
  --ckpt "${OUT_DIR}" \
  --override "logging.save_dir=${OUT_DIR}/eval" \
  --override "logging.metrics_path=${OUT_DIR}/eval/metrics.jsonl" \
  --override "logging.final_metrics_path=${OUT_DIR}/eval/metrics.json" \
  --override "logging.matrix_path=${OUT_DIR}/eval/matrix.json" \
  --override "logging.backend=wandb" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.group=${WANDB_GROUP}_eval" \
  --override "logging.wandb.mode=${WANDB_MODE}" \
  --override "logging.wandb.name=${MODEL_TAG}/${RUN_NAME}/budget_${BUDGET}/seed${SEED}/eval" \
  > "${EVAL_LOG}" 2>&1

echo "[done] ${OUT_DIR}"

