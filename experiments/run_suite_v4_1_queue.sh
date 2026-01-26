#!/usr/bin/env bash
set -euo pipefail

# Queue launcher: keep ALL GPUs busy, but NEVER run 2 big-model trainings on the same GPU
# (that caused OOM when ARC+GSM8K were colocated).
#
# Suite versions:
# - ARC:  v4_1  (smart_spectral_guided, budget_large)
# - GSM8K: v2_1 (smart_spectral_guided)
#
# Models: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B
#
# Usage:
#   nohup bash experiments/run_suite_v4_1_queue.sh > results/_nohup_v4_1_queue.log 2>&1 &

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

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"

LORA_TARGET='["q_proj","k_proj","v_proj","o_proj"]'
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

ALPHA="${ALPHA:-0.01}"
GAMMA="${GAMMA:-0.2}"
TAU="${TAU:-0.1}"

ARC_BS="${ARC_BS:-1}"
ARC_MAXLEN="${ARC_MAXLEN:-256}"
GSM_BS="${GSM_BS:-1}"
GSM_MAXLEN="${GSM_MAXLEN:-512}"

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

run_arc() {
  local gpu="$1"
  local model="$2"
  local tag
  tag="$(tagify "$model")"
  local root="results/arc/${tag}_v4_1"
  local out="${root}/smart_spectral_guided/budget_large/seed42"
  local log="${root}/_logs/smart_spectral_guided.log"
  mkdir -p "${root}/_logs" "${out}"

  if is_done_json "${out}/metrics.json"; then
    echo "[skip][arc] already done: ${out}"
    return 0
  fi

  echo "[run][arc] gpu=${gpu} model=${model} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m experiments.train_classification \
    --config configs/classification/arc/erm.yaml \
    --override "model_name_or_path=${model}" \
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
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/metrics.json" \
    > "${log}" 2>&1

  mkdir -p "${out}/eval"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m experiments.eval_classification \
    --config "${out}/config_resolved.yaml" \
    --ckpt "${out}" \
    --override "logging.save_dir=${out}/eval" \
    --override "logging.final_metrics_path=${out}/eval/metrics.json" \
    >> "${log}" 2>&1
}

run_gsm8k() {
  local gpu="$1"
  local model="$2"
  local tag
  tag="$(tagify "$model")"
  local root="results/gsm8k/${tag}_gsm8k_suite_v2_1"
  local out="${root}/smart_spectral_guided/seed42"
  local log="${root}/_logs/smart_spectral_guided.log"
  mkdir -p "${root}/_logs" "${out}"

  if is_done_json "${out}/metrics.json"; then
    echo "[skip][gsm8k] already done: ${out}"
    return 0
  fi

  echo "[run][gsm8k] gpu=${gpu} model=${model} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m experiments.train_generation \
    --config configs/generation/gsm8k/nll.yaml \
    --override "model_name_or_path=${model}" \
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
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/metrics.json" \
    > "${log}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" python -m experiments.eval_generation \
    --config "configs/generation/gsm8k/nll.yaml" \
    --ckpt "${out}" \
    --override "model_name_or_path=${model}" \
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/eval_metrics_v2.json" \
    >> "${log}" 2>&1
}

# Job list: 6 jobs total; scheduler assigns them round-robin to GPUs, 1 job per GPU at a time.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

JOBS=()
for m in "${MODELS[@]}"; do JOBS+=("arc|${m}"); done
for m in "${MODELS[@]}"; do JOBS+=("gsm8k|${m}"); done

declare -A PID_BY_GPU
job_idx=0

for spec in "${JOBS[@]}"; do
  IFS='|' read -r kind model <<< "${spec}"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"

  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    wait "${prev}" || true
    unset PID_BY_GPU["$gpu"]
  fi

  if [[ "${kind}" == "arc" ]]; then
    (run_arc "${gpu}" "${model}") &
  else
    (run_gsm8k "${gpu}" "${model}") &
  fi

  PID_BY_GPU["$gpu"]=$!
  job_idx=$((job_idx + 1))
  sleep 2
done

echo "[wait] waiting for remaining jobs..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
echo "[done] all queued jobs finished."

