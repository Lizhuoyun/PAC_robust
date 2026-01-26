#!/usr/bin/env bash
set -euo pipefail

# Experiment: Spectral-Guided SMART (S-SMART)
# Model: Qwen2.5-0.5B-Instruct
# Task: ARC (Classification)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
source "${VENV}/bin/activate"

export HF_HOME="/LOCAL2/zhuoyun/hf_cache"
export NLTK_DATA="/LOCAL2/zhuoyun/nltk_data"

GPU=2
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SEED=42
LORA_TARGET='["q_proj","k_proj","v_proj","o_proj"]'

# 1. Baseline: SMART only (for reference)
OUT_BASE="results/arc/qwen05b_guided_smoke/smart_only"
mkdir -p "${OUT_BASE}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.train_classification \
  --config configs/classification/arc/erm.yaml \
  --override "model_name_or_path=${MODEL}" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "smart.enabled=true" \
  --override "spectral.enabled=false" \
  --override "logging.save_dir=${OUT_BASE}" \
  --override "logging.final_metrics_path=${OUT_BASE}/metrics.json"

# 2. Parallel: SMART + Spectral (External parallel)
OUT_PARA="results/arc/qwen05b_guided_smoke/smart_plus_spectral"
mkdir -p "${OUT_PARA}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.train_classification \
  --config configs/classification/arc/erm.yaml \
  --override "model_name_or_path=${MODEL}" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "smart.enabled=true" \
  --override "spectral.enabled=true" \
  --override "spectral.alpha=0.01" \
  --override "logging.save_dir=${OUT_PARA}" \
  --override "logging.final_metrics_path=${OUT_PARA}/metrics.json"

# 3. Guided: Spectral-Guided SMART (Internal coupling)
OUT_GUIDED="results/arc/qwen05b_guided_smoke/smart_spectral_guided"
mkdir -p "${OUT_GUIDED}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.train_classification \
  --config configs/classification/arc/erm.yaml \
  --override "model_name_or_path=${MODEL}" \
  --override "lora.target_modules=${LORA_TARGET}" \
  --override "smart.enabled=true" \
  --override "smart.spectral_guided=true" \
  --override "spectral.enabled=true" \
  --override "spectral.inner_guided=true" \
  --override "spectral.alpha=0.01" \
  --override "logging.save_dir=${OUT_GUIDED}" \
  --override "logging.final_metrics_path=${OUT_GUIDED}/metrics.json"

# --- Run Evaluation for all three ---
for d in "${OUT_BASE}" "${OUT_PARA}" "${OUT_GUIDED}"; do
  echo "[eval] $d"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.eval_classification \
    --config "${d}/config_resolved.yaml" \
    --ckpt "$d" \
    --override "logging.save_dir=${d}/eval" \
    --override "logging.final_metrics_path=${d}/eval/metrics.json"
done

echo "All smoke tests completed. Check results under results/arc/qwen05b_guided_smoke/"
