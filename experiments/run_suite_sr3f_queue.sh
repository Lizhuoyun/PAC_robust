#!/usr/bin/env bash
set -euo pipefail

# S-R3F: Spectral inside R3F (spectral loss on the same noisy point as R3F).
# Runs 3 models x ARC + GSM8K = 6 jobs, 1 per GPU at a time.
#
# Key constraints:
# - Do NOT overwrite existing results (write to new versioned roots).
# - Keep GPUs saturated but stable: 1 big-model training per GPU at a time (queue scheduler).
#
# Output roots:
# - ARC:   results/arc/<model_tag>_v4_sr3f/r3f_spectral_guided/budget_large/seed42
# - GSM8K: results/gsm8k/<model_tag>_gsm8k_suite_v2_sr3f/r3f_spectral_guided/seed42
#
# Optional env:
#   GPUS="0,1,2"
#   WANDB_PROJECT=icml_wcr_spectral_v4
#   WANDB_MODE=online|offline|disabled
#   ALPHA=0.01
#   ARC_BS=1  GSM_BS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="${VENV:-/LOCAL2/zhuoyun/Robustfairnessgpu3/venv}"
source "${VENV}/bin/activate"
PYTHON="${PYTHON:-${VENV}/bin/python}"

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

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

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

wait_gpu_free() {
  local gpu="$1"  # physical index
  local dev="/dev/nvidia${gpu}"
  # If the device node doesn't exist, just don't block (some clusters hide /dev/nvidia*).
  if [[ ! -e "${dev}" ]]; then
    return 0
  fi
  # Wait until no processes are using the GPU device file.
  while fuser "${dev}" >/dev/null 2>&1; do
    echo "[wait][gpu${gpu}] busy (fuser ${dev}). sleeping 30s..."
    sleep 30
  done
}

with_gpu_lock() {
  local gpu="$1"; shift
  local lock="${ROOT_DIR}/results/_gpu_lock_${gpu}.lock"
  mkdir -p "${ROOT_DIR}/results"
  (
    exec 9>"${lock}"
    flock 9
    wait_gpu_free "${gpu}"
    "$@"
  )
}

run_arc() {
  local gpu="$1"
  local model="$2"
  local tag; tag="$(tagify "$model")"
  local root="results/arc/${tag}_v4_sr3f"
  local out="${root}/r3f_spectral_guided/budget_large/seed42"
  local log="${root}/_logs/r3f_spectral_guided.log"
  mkdir -p "${root}/_logs" "${out}"

  if is_done_json "${out}/metrics.json"; then
    echo "[skip][arc] already done: ${out}"
    return 0
  fi

  echo "[run][arc] gpu=${gpu} model=${model} S-R3F -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m experiments.train_classification \
    --config configs/classification/arc/erm.yaml \
    --override "seed=42" \
    --override "model_name_or_path=${model}" \
    --override "batch_size=${ARC_BS}" \
    --override "max_length=${ARC_MAXLEN}" \
    --override "model.torch_dtype=${MODEL_DTYPE}" \
    --override "lora.target_modules=${LORA_TARGET}" \
    --override "augment.enabled=false" \
    --override "r3f.enabled=true" \
    --override "r3f.spectral_guided=true" \
    --override "smart.enabled=false" \
    --override "spectral.enabled=false" \
    --override "spectral.alpha=${ALPHA}" \
    --override "spectral.gamma=${GAMMA}" \
    --override "spectral.tau=${TAU}" \
    --override "spectral.stability=false" \
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/metrics.json" \
    > "${log}" 2>&1

  mkdir -p "${out}/eval"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m experiments.eval_classification \
    --config "${out}/config_resolved.yaml" \
    --ckpt "${out}" \
    --override "logging.save_dir=${out}/eval" \
    --override "logging.final_metrics_path=${out}/eval/metrics.json" \
    >> "${log}" 2>&1
}

run_gsm8k() {
  local gpu="$1"
  local model="$2"
  local tag; tag="$(tagify "$model")"
  local root="results/gsm8k/${tag}_gsm8k_suite_v2_sr3f"
  local out="${root}/r3f_spectral_guided/seed42"
  local log="${root}/_logs/r3f_spectral_guided.log"
  mkdir -p "${root}/_logs" "${out}"

  if is_done_json "${out}/metrics.json"; then
    echo "[skip][gsm8k] already done: ${out}"
    return 0
  fi

  echo "[run][gsm8k] gpu=${gpu} model=${model} S-R3F -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m experiments.train_generation \
    --config configs/generation/gsm8k/nll.yaml \
    --override "seed=42" \
    --override "model_name_or_path=${model}" \
    --override "batch_size=${GSM_BS}" \
    --override "max_length=${GSM_MAXLEN}" \
    --override "model.torch_dtype=${MODEL_DTYPE}" \
    --override "lora.target_modules=${LORA_TARGET}" \
    --override "augment.enabled=false" \
    --override "r3f.enabled=true" \
    --override "r3f.spectral_guided=true" \
    --override "smart.enabled=false" \
    --override "spectral.enabled=false" \
    --override "spectral.alpha=${ALPHA}" \
    --override "spectral.gamma=${GAMMA}" \
    --override "spectral.tau=${TAU}" \
    --override "spectral.stability=false" \
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/metrics.json" \
    > "${log}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m experiments.eval_generation \
    --config "configs/generation/gsm8k/nll.yaml" \
    --ckpt "${out}" \
    --override "model_name_or_path=${model}" \
    --override "logging.save_dir=${out}" \
    --override "logging.final_metrics_path=${out}/eval_metrics_v2.json" \
    >> "${log}" 2>&1
}

# Build job list: 3 models * 2 datasets = 6 jobs
JOBS=()
for m in "${MODELS[@]}"; do JOBS+=("arc|${m}"); done
for m in "${MODELS[@]}"; do JOBS+=("gsm8k|${m}"); done

# One-job-per-GPU scheduler
declare -A PID_BY_GPU
declare -A SPEC_BY_PID
FAILURES=0
job_idx=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r kind model <<< "${spec}"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"

  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    if ! wait "${prev}"; then
      echo "[fail] job failed on gpu=${gpu}: ${SPEC_BY_PID[$prev]:-unknown}"
      FAILURES=$((FAILURES + 1))
    fi
    unset PID_BY_GPU["$gpu"]
  fi

  if [[ "${kind}" == "arc" ]]; then
    (with_gpu_lock "${gpu}" run_arc "${gpu}" "${model}") &
  else
    (with_gpu_lock "${gpu}" run_gsm8k "${gpu}" "${model}") &
  fi

  pid=$!
  PID_BY_GPU["$gpu"]=$pid
  SPEC_BY_PID["$pid"]="${spec}"
  job_idx=$((job_idx + 1))
  sleep 2
done

echo "[wait] waiting for remaining jobs..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    if ! wait "${pid}"; then
      echo "[fail] job failed on gpu=${gpu}: ${SPEC_BY_PID[$pid]:-unknown}"
      FAILURES=$((FAILURES + 1))
    fi
  fi
done
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "[done-with-failures] finished with ${FAILURES} failed job(s)."
  exit 1
fi
echo "[done] all S-R3F jobs finished."

