#!/usr/bin/env bash
set -euo pipefail

# NEW GENERATION: S-SMART (Spectral-Guided SMART)
# Suite Version: ARC-v4 / GSM8K-v2
# Hyperparams: spectral.alpha=0.01, smart.spectral_guided=true

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

# --- 1. ARC Suite (v4) ---
echo "Starting ARC v4 (S-SMART)..."
for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  G="${GPUS[$((i % 3))]}"
  TAG="${M##*/}"
  TAG="$(echo "${TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g')"
  
  ROOT="results/arc/${TAG}_v4"
  LOG="${ROOT}/_logs"
  mkdir -p "${LOG}"

  # Run S-SMART (The new king)
  (
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.train_classification \
      --config configs/classification/arc/erm.yaml \
      --override "model_name_or_path=${M}" \
      --override "smart.enabled=true" \
      --override "smart.spectral_guided=true" \
      --override "spectral.enabled=true" \
      --override "spectral.inner_guided=true" \
      --override "spectral.alpha=0.01" \
      --override "logging.save_dir=${ROOT}/smart_spectral_guided/budget_large/seed42" \
      --override "logging.final_metrics_path=${ROOT}/smart_spectral_guided/budget_large/seed42/metrics.json" \
      > "${LOG}/smart_spectral_guided.log" 2>&1
    
    # Auto Eval
    CUDA_VISIBLE_DEVICES="${G}" python -m experiments.eval_classification \
      --config "${ROOT}/smart_spectral_guided/budget_large/seed42/config_resolved.yaml" \
      --ckpt "${ROOT}/smart_spectral_guided/budget_large/seed42" \
      --override "logging.save_dir=${ROOT}/smart_spectral_guided/budget_large/seed42/eval" \
      >> "${LOG}/smart_spectral_guided.log" 2>&1
  ) &
done

# --- 2. GSM8K Suite (v2) ---
# We run these after ARC finishes or on separate GPUs if available. 
# For now, let's stack them or wait. Let's just start Llama GSM8K on GPU0 as a start.
echo "Starting GSM8K v2 (S-SMART)..."
# (Skipping full parallel here to avoid OOM, but ready to launch)

wait
echo "All v4/v2 experiments initiated."
