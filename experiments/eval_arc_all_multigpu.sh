#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU evaluator for ARC (classification).
# It scans one or more results roots for completed LoRA checkpoints and runs evaluation.
#
# Default roots:
#   results/arc/qwen7b_full_mix111
#   results/arc/llama31_8b_full_mix111
#   results/arc/mistral7b_full_mix111
#
# Outputs are written to: <ckpt_dir>/eval/
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=arc_eval_all
#   WANDB_MODE=online
#   GPUS="0,1,2"
#   ROOTS="results/arc/qwen7b_full_mix111 results/arc/llama31_8b_full_mix111"
#

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
WANDB_GROUP="${WANDB_GROUP:-arc_eval_all}"
WANDB_MODE="${WANDB_MODE:-online}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified (GPUS=${GPUS_RAW})"
  exit 1
fi

ROOTS_STR="${ROOTS:-results/arc/qwen7b_full_mix111 results/arc/llama31_8b_full_mix111 results/arc/mistral7b_full_mix111}"
read -r -a ROOTS_ARR <<< "${ROOTS_STR}"

LOG_ROOT="results/arc/_eval_logs"
mkdir -p "${LOG_ROOT}"

MAX_JOBS="${MAX_JOBS:-0}"  # 0 = no limit

is_completed_ckpt() {
  local ckpt_dir="$1"
  [[ -f "${ckpt_dir}/adapter_config.json" ]] || return 1
  [[ -f "${ckpt_dir}/config_resolved.yaml" ]] || return 1
  [[ -f "${ckpt_dir}/metrics.json" ]] || return 1
  grep -q "\"status\"\\s*:\\s*\"done\"" "${ckpt_dir}/metrics.json" || return 1
  return 0
}

run_one() {
  local gpu="$1"
  local ckpt_dir="$2"
  local root_tag="$3"
  local root_root="$4"

  local eval_dir="${ckpt_dir}/eval"
  local ckpt_id
  ckpt_id="$(echo -n "${ckpt_dir}" | md5sum | awk '{print $1}')"
  local log_file="${LOG_ROOT}/${root_tag}__${ckpt_id}__gpu${gpu}.log"
  mkdir -p "${eval_dir}"

  # Relative path used for W&B run naming.
  local rel="${ckpt_dir#${root_root}/}"
  if [[ "${rel}" == "${ckpt_dir}" ]]; then
    rel="${ckpt_dir##*/}"
  fi

  echo "[eval] gpu=${gpu} suite=${root_tag} ckpt=${ckpt_dir} log=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m experiments.eval_classification \
    --config "${ckpt_dir}/config_resolved.yaml" \
    --ckpt "${ckpt_dir}" \
    --override "logging.save_dir=${eval_dir}" \
    --override "logging.metrics_path=${eval_dir}/metrics.jsonl" \
    --override "logging.final_metrics_path=${eval_dir}/metrics.json" \
    --override "logging.matrix_path=${eval_dir}/matrix.json" \
    --override "logging.backend=wandb" \
    --override "logging.wandb.project=${WANDB_PROJECT}" \
    --override "logging.wandb.group=${WANDB_GROUP}" \
    --override "logging.wandb.mode=${WANDB_MODE}" \
    --override "logging.wandb.name=${root_tag}/eval/${rel}" \
    > "${log_file}" 2>&1
}

# Build ckpt list (only completed ones).
CKPTS=()
for root in "${ROOTS_ARR[@]}"; do
  if [[ ! -d "${root}" ]]; then
    echo "[skip] root not found: ${root}"
    continue
  fi
  while IFS= read -r dir; do
    if is_completed_ckpt "${dir}"; then
      CKPTS+=("${dir}")
    fi
  done < <(find "${root}" -type f -name adapter_config.json -printf '%h\n' | sort -u)
done

echo "[info] found ${#CKPTS[@]} completed checkpoints to evaluate"
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  exit 0
fi

# One-job-per-GPU scheduler.
declare -A PID_BY_GPU
job_idx=0

for ckpt in "${CKPTS[@]}"; do
  # suite name under results/arc, e.g. qwen7b_full_mix111
  suite="${ckpt#results/arc/}"
  suite="${suite%%/*}"
  root_tag="${suite}"
  root_root="results/arc/${suite}"

  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    wait "${prev}" || true
    unset PID_BY_GPU["$gpu"]
  fi
  (run_one "${gpu}" "${ckpt}" "${root_tag}" "${root_root}") &
  PID_BY_GPU["$gpu"]=$!
  job_idx=$((job_idx + 1))
  sleep 1
  if [[ "${MAX_JOBS}" != "0" && "${job_idx}" -ge "${MAX_JOBS}" ]]; then
    echo "[info] MAX_JOBS reached (${MAX_JOBS}); stopping new launches"
    break
  fi
done

echo "[wait] waiting for remaining eval jobs..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
echo "[done] all eval jobs completed."


