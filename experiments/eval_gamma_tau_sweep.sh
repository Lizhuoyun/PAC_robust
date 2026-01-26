#!/usr/bin/env bash
set -euo pipefail

# Eval-only sweep for spectral.gamma / spectral.tau.
#
# IMPORTANT:
# - This sweep DOES NOT change model predictions, so clean_acc / robust_acc won't change.
# - It DOES change matrix_gamma -> wcr_* and sigma_max_* metrics.
#
# Usage:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv \
#   CKPT=results/arc/qwen7b_tuned_v3/erm_smart_spectral/budget_large/seed42 \
#   GPUS=0,1,2 \
#   GAMMAS="0.0 0.05 0.1 0.2 0.3" \
#   TAUS="0.05 0.1 0.2 0.4" \
#   WANDB_PROJECT=icml_wcr_spectral \
#   WANDB_GROUP=qwen7b_tuned_v3_gamma_tau_eval_sweep_v1 \
#   bash experiments/eval_gamma_tau_sweep.sh

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

CKPT="${CKPT:-results/arc/qwen7b_tuned_v3/erm_smart_spectral/budget_large/seed42}"
CFG="${CFG:-${CKPT}/config_resolved.yaml}"
if [[ ! -f "${CFG}" ]]; then
  echo "config not found: ${CFG}"
  exit 1
fi
if [[ ! -d "${CKPT}" ]]; then
  echo "ckpt dir not found: ${CKPT}"
  exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-icml_wcr_spectral}"
WANDB_GROUP="${WANDB_GROUP:-qwen7b_tuned_v3_gamma_tau_eval_sweep_v1}"
WANDB_MODE="${WANDB_MODE:-online}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified (GPUS=${GPUS_RAW})"
  exit 1
fi

GAMMAS_STR="${GAMMAS:-0.0 0.05 0.1 0.2 0.3}"
TAUS_STR="${TAUS:-0.05 0.1 0.2 0.4}"
read -r -a GAMMAS_ARR <<< "${GAMMAS_STR}"
read -r -a TAUS_ARR <<< "${TAUS_STR}"

OUT_ROOT="${OUT_ROOT:-${CKPT}/eval_gamma_tau}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/_logs}"
mkdir -p "${LOG_ROOT}"

CKPT_TAG="${CKPT#results/arc/}"
CKPT_TAG="$(echo "${CKPT_TAG}" | sed -E 's/[^a-zA-Z0-9._+-]+/_/g')"

run_one() {
  local gpu="$1"
  local gamma="$2"
  local tau="$3"

  local name="g${gamma}_t${tau}"
  local out_dir="${OUT_ROOT}/${name}"
  local log_file="${LOG_ROOT}/${name}__gpu${gpu}.log"
  mkdir -p "${out_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m experiments.eval_classification \
    --config "${CFG}" \
    --ckpt "${CKPT}" \
    --override "spectral.gamma=${gamma}" \
    --override "spectral.tau=${tau}" \
    --override "logging.backend=wandb" \
    --override "logging.wandb.project=${WANDB_PROJECT}" \
    --override "logging.wandb.group=${WANDB_GROUP}" \
    --override "logging.wandb.mode=${WANDB_MODE}" \
    --override "logging.wandb.name=${CKPT_TAG}/eval_gamma_tau/${name}" \
    --override "logging.save_dir=${out_dir}" \
    --override "logging.metrics_path=${out_dir}/metrics.jsonl" \
    --override "logging.final_metrics_path=${out_dir}/metrics.json" \
    --override "logging.matrix_path=${out_dir}/matrix.json" \
    > "${log_file}" 2>&1
}

declare -A PID_BY_GPU
job_idx=0

for gamma in "${GAMMAS_ARR[@]}"; do
  for tau in "${TAUS_ARR[@]}"; do
    gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
    prev="${PID_BY_GPU[$gpu]:-}"
    if [[ -n "${prev}" ]]; then
      wait "${prev}" || true
      unset PID_BY_GPU["$gpu"]
    fi
    (run_one "${gpu}" "${gamma}" "${tau}") &
    PID_BY_GPU["$gpu"]=$!
    job_idx=$((job_idx + 1))
    sleep 1
  done
done

echo "[wait] waiting for remaining eval jobs..."
for gpu in "${GPU_LIST[@]}"; do
  pid="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
echo "[done] sweep completed: ${OUT_ROOT}"



