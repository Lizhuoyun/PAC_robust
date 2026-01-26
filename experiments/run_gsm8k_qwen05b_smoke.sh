#!/usr/bin/env bash
set -euo pipefail

# Smoke test: run GSM8K generation (NLL + LoRA) on Qwen 0.5B and evaluate.
# Goal: validate training/eval pipeline + wandb logging + outputs.
#
# Usage:
#   bash experiments/run_gsm8k_qwen05b_smoke.sh
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=gsm8k_qwen05b_smoke
#   WANDB_MODE=online   (or offline/disabled)
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
WANDB_GROUP="${WANDB_GROUP:-gsm8k_qwen05b_smoke}"
WANDB_MODE="${WANDB_MODE:-online}"

GPU="${GPU:-0}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
LORA_TARGET="${LORA_TARGET_MODULES:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]}"
SEED="${SEED:-42}"

CONFIG="${CONFIG_PATH:-configs/generation/gsm8k/nll.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gsm8k/qwen05b_smoke}"
OUT_DIR="${RESULTS_ROOT}/erm/seed${SEED}"
mkdir -p "${OUT_DIR}"

echo "[smoke] gpu=${GPU} model=${MODEL} -> ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
python -m experiments.train_generation \
  --config "${CONFIG}" \
  --override "seed=${SEED}" \
  --override "model_name_or_path=${MODEL}" \
  --override "model.torch_dtype=bf16" \
  --override "finetune_mode=lora" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "lora.r=8" \
  --override "max_length=512" \
  --override "batch_size=8" \
  --override "epochs=1" \
  --override "dataset.max_train_examples=200" \
  --override "dataset.max_eval_examples=64" \
  --override "logging.backend=both" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.name=qwen05b/erm/seed${SEED}/train" \
  --override "logging.wandb.mode=${WANDB_MODE}" \
  --override "logging.save_dir=${OUT_DIR}" \
  --override "logging.metrics_path=${OUT_DIR}/train_metrics.jsonl" \
  --override "logging.final_metrics_path=${OUT_DIR}/train_metrics.json"

CUDA_VISIBLE_DEVICES="${GPU}" \
python -m experiments.eval_generation \
  --config "${CONFIG}" \
  --ckpt "${OUT_DIR}" \
  --override "seed=${SEED}" \
  --override "model_name_or_path=${MODEL}" \
  --override "model.torch_dtype=bf16" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "dataset.max_eval_examples=64" \
  --override "logging.backend=both" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.name=qwen05b/erm/seed${SEED}/eval" \
  --override "logging.wandb.mode=${WANDB_MODE}" \
  --override "logging.save_dir=${OUT_DIR}" \
  --override "logging.metrics_path=${OUT_DIR}/eval_metrics.jsonl" \
  --override "logging.final_metrics_path=${OUT_DIR}/eval_metrics.json"

echo "[smoke][done] ${OUT_DIR}"


