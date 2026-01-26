#!/usr/bin/env bash
set -euo pipefail

# S-SMART (Spectral-Guided SMART) staggered launcher
# Goal: avoid CPU/RAM OOM during simultaneous model shard loading.
#
# Strategy:
# - Launch ARC v4 (3 models, 1 per GPU) first.
# - Wait until each GPU shows substantial memory usage (model loaded).
# - Then launch GSM8K v2 (3 models, 1 per GPU).
#
# This keeps GPU utilization high while preventing "Terminated" due to RAM spikes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
source "${VENV}/bin/activate"

export HF_HOME="${HF_HOME:-/LOCAL2/zhuoyun/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${NLTK_DATA:-/LOCAL2/zhuoyun/nltk_data}"

export WANDB_PROJECT="${WANDB_PROJECT:-icml_wcr_spectral_v4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
GPUS=(0 1 2)
LORA_TARGET='["q_proj","k_proj","v_proj","o_proj"]'

ALPHA="${ALPHA:-0.01}"
GAMMA="${GAMMA:-0.2}"
TAU="${TAU:-0.1}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
ARC_BS="${ARC_BS:-1}"
ARC_MAXLEN="${ARC_MAXLEN:-256}"
GSM_BS="${GSM_BS:-1}"
GSM_MAXLEN="${GSM_MAXLEN:-512}"

GPU_READY_MEM_MIB="${GPU_READY_MEM_MIB:-8000}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-900}"

kill_old() {
  pkill -f "python -m experiments.train_classification" || true
  pkill -f "python -m experiments.train_generation" || true
}

gpu_mem_used_mib() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_for_gpus_ready() {
  local start
  start="$(date +%s)"
  while true; do
    local ready=0
    for gpu in "${GPUS[@]}"; do
      mem="$(gpu_mem_used_mib "${gpu}")"
      if [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem >= GPU_READY_MEM_MIB )); then
        ready=$((ready + 1))
      fi
    done
    if (( ready == ${#GPUS[@]} )); then
      echo "[ready] all GPUs have >= ${GPU_READY_MEM_MIB} MiB allocated"
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start > MAX_WAIT_SEC )); then
      echo "[warn] timeout waiting for GPU mem >= ${GPU_READY_MEM_MIB} MiB (ready=${ready}/${#GPUS[@]})"
      return 1
    fi
    sleep 5
  done
}

tagify() {
  local m="$1"
  local tag="${m##*/}"
  echo "${tag}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._+]+/_/g'
}

launch_arc() {
  for i in "${!MODELS[@]}"; do
    M="${MODELS[$i]}"
    G="${GPUS[$i]}"
    TAG="$(tagify "${M}")"
    ROOT_ARC="results/arc/${TAG}_v4_1"
    mkdir -p "${ROOT_ARC}/_logs"
    (
      echo "[arc] gpu=${G} model=${M}"
      CUDA_VISIBLE_DEVICES="${G}" python -m experiments.train_classification \
        --config configs/classification/arc/erm.yaml \
        --override "model_name_or_path=${M}" \
        --override "batch_size=${ARC_BS}" \
        --override "max_length=${ARC_MAXLEN}" \
        --override "model.torch_dtype=${MODEL_DTYPE}" \
        --override "lora.target_modules=${LORA_TARGET}" \
        --override "smart.enabled=true" \
        --override "smart.spectral_guided=true" \
        --override "spectral.enabled=true" \
        --override "spectral.inner_guided=true" \
        --override "spectral.alpha=${ALPHA}" \
        --override "spectral.gamma=${GAMMA}" \
        --override "spectral.tau=${TAU}" \
        --override "spectral.stability=false" \
        --override "logging.save_dir=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42" \
        --override "logging.final_metrics_path=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/metrics.json" \
        > "${ROOT_ARC}/_logs/smart_spectral_guided.log" 2>&1

      CUDA_VISIBLE_DEVICES="${G}" python -m experiments.eval_classification \
        --config "${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/config_resolved.yaml" \
        --ckpt "${ROOT_ARC}/smart_spectral_guided/budget_large/seed42" \
        --override "logging.save_dir=${ROOT_ARC}/smart_spectral_guided/budget_large/seed42/eval" \
        >> "${ROOT_ARC}/_logs/smart_spectral_guided.log" 2>&1
    ) &
    sleep 2
  done
}

launch_gsm8k() {
  for i in "${!MODELS[@]}"; do
    M="${MODELS[$i]}"
    G="${GPUS[$i]}"
    TAG="$(tagify "${M}")"
    ROOT_GSM="results/gsm8k/${TAG}_gsm8k_suite_v2_1"
    mkdir -p "${ROOT_GSM}/_logs"
    (
      echo "[gsm8k] gpu=${G} model=${M}"
      CUDA_VISIBLE_DEVICES="${G}" python -m experiments.train_generation \
        --config configs/generation/gsm8k/nll.yaml \
        --override "model_name_or_path=${M}" \
        --override "batch_size=${GSM_BS}" \
        --override "max_length=${GSM_MAXLEN}" \
        --override "model.torch_dtype=${MODEL_DTYPE}" \
        --override "lora.target_modules=${LORA_TARGET}" \
        --override "smart.enabled=true" \
        --override "smart.spectral_guided=true" \
        --override "spectral.enabled=true" \
        --override "spectral.inner_guided=true" \
        --override "spectral.alpha=${ALPHA}" \
        --override "spectral.gamma=${GAMMA}" \
        --override "spectral.tau=${TAU}" \
        --override "spectral.stability=false" \
        --override "logging.save_dir=${ROOT_GSM}/smart_spectral_guided/seed42" \
        --override "logging.final_metrics_path=${ROOT_GSM}/smart_spectral_guided/seed42/metrics.json" \
        > "${ROOT_GSM}/_logs/smart_spectral_guided.log" 2>&1

      CUDA_VISIBLE_DEVICES="${G}" python -m experiments.eval_generation \
        --config "configs/generation/gsm8k/nll.yaml" \
        --ckpt "${ROOT_GSM}/smart_spectral_guided/seed42" \
        --override "model_name_or_path=${M}" \
        --override "logging.save_dir=${ROOT_GSM}/smart_spectral_guided/seed42" \
        --override "logging.final_metrics_path=${ROOT_GSM}/smart_spectral_guided/seed42/eval_metrics_v2.json" \
        >> "${ROOT_GSM}/_logs/smart_spectral_guided.log" 2>&1
    ) &
    sleep 2
  done
}

kill_old
echo "[launch] ARC v4 first..."
launch_arc
echo "[wait] for GPUs to allocate memory (avoid RAM spike on simultaneous loads)..."
wait_for_gpus_ready || true
echo "[launch] GSM8K v2 next..."
launch_gsm8k

echo "[done] launched all jobs; tail logs under results/arc/*_v4/_logs and results/gsm8k/*_suite_v2/_logs"
wait

