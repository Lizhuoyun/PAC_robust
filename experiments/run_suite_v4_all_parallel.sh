#!/usr/bin/env bash
set -euo pipefail

# S-SMART (Spectral-Adversarial Training)
# MAXIMIZE GPU UTILIZATION: ARC v4 + GSM8K v2 in Parallel
# Models: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
source "${VENV}/bin/activate"

export HF_HOME="/LOCAL2/zhuoyun/hf_cache"
export NLTK_DATA="/LOCAL2/zhuoyun/nltk_data"
export WANDB_PROJECT="icml_wcr_spectral_v4"
export WANDB_MODE="online"

MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
GPUS=(0 1 2)
LORA_TARGET='["q_proj","k_proj","v_proj","o_proj"]'

# Cleanup old failed runs
pkill -f "train_classification" || true
pkill -f "train_generation" || true

echo "Launching ARC v4 and GSM8K v2 in parallel (2 jobs per GPU)..."

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  G="${GPUS[$i]}"
  TAG="${M##*/}"
  TAG="$(echo "${TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g')"
  
  # --- ARC Job ---
  ROOT_ARC="results/arc/${TAG}_v4"
  mkdir -p "${ROOT_ARC}/_logs"
  (
    echo "[arc] gpu=${G} model=${M}"
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.train_classification \
      --config configs/classification/arc/erm.yaml \
      --override "model_name_or_path=${M}" \
      --override "lora.target_modules=${LORA_TARGET}" \
      --override "smart.enabled=true" \
      --override "smart.spectral_guided=true" \
      --override "spectral.enabled=true" \
      --override "spectral.inner_guided=true" \
      --override "spectral.alpha=0.01" \
      --override "logging.save_dir=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42" \
      --override "logging.final_metrics_path=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/metrics.json" \
      > "${ROOT_ARC}/_logs/smart_spectral_guided.log" 2>&1
    
    # Auto Eval ARC
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.eval_classification \
      --config "${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/config_resolved.yaml" \
      --ckpt "${ROOT_ARC}/smart_spectral_guided/budget_large/seed42" \
      --override "logging.save_dir=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/eval" \
      >> "${ROOT_ARC}/_logs/smart_spectral_guided.log" 2>&1
  ) &

  # --- GSM8K Job ---
  ROOT_GSM="results/gsm8k/${TAG}_gsm8k_suite_v2"
  mkdir -p "${ROOT_GSM}/_logs"
  (
    echo "[gsm8k] gpu=${G} model=${M}"
    # Using nll.yaml as config for generation
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.train_generation \
      --config configs/generation/gsm8k/nll.yaml \
      --override "model_name_or_path=${M}" \
      --override "lora.target_modules=${LORA_TARGET}" \
      --override "smart.enabled=true" \
      --override "smart.spectral_guided=true" \
      --override "spectral.enabled=true" \
      --override "spectral.inner_guided=true" \
      --override "spectral.alpha=0.01" \
      --override "logging.save_dir=${ROOT_GSM}/smart_spectral_guided/seed42" \
      --override "logging.final_metrics_path=${ROOT_GSM}/smart_spectral_guided/seed42/metrics.json" \
      > "${ROOT_GSM}/_logs/smart_spectral_guided.log" 2>&1
    
    # Auto Eval GSM8K
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.eval_generation \
      --config "configs/generation/gsm8k/nll.yaml" \
      --ckpt "${ROOT_GSM}/smart_spectral_guided/seed42" \
      --override "model_name_or_path=${M}" \
      --override "logging.save_dir=${ROOT_GSM}/smart_spectral_guided/seed42" \
      --override "logging.final_metrics_path=${ROOT_GSM}/smart_spectral_guided/seed42/eval_metrics_v2.json" \
      >> "${ROOT_GSM}/_logs/smart_spectral_guided.log" 2>&1
  ) &
done

echo "All jobs launched. Monitor with nvidia-smi."
wait
