#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU evaluator for ARC (classification) with multiple *evaluation profiles*.
#
# Motivation:
# - Make robustness evaluation more convincing by testing distribution shift:
#   mix111 (ID) vs type-only (OOD) vs stress settings.
# - Works for any results root that contains LoRA checkpoints with metrics.json status=done.
#
# Outputs:
#   <ckpt_dir>/eval_profiles/<profile_name>/{metrics.jsonl,metrics.json,matrix.json,...}
#
# Optional env:
#   VENV=/LOCAL2/zhuoyun/Robustfairnessgpu3/venv
#   HF_HOME=/LOCAL2/zhuoyun/hf_cache
#   WANDB_PROJECT=icml_wcr_spectral
#   WANDB_GROUP=qwen7b_tuned_v3_eval_profiles_v1
#   WANDB_MODE=online
#   GPUS="0,1,2"
#   ROOTS="results/arc/qwen7b_tuned_v3"
#   PROFILES="mix111 typo_only synonym_only paraphrase_only typo_stress synonym_stress paraphrase_stress"
#   MAX_JOBS=0   # 0 = no limit

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
WANDB_GROUP="${WANDB_GROUP:-qwen7b_tuned_v3_eval_profiles_v1}"
WANDB_MODE="${WANDB_MODE:-online}"

GPUS_RAW="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_RAW}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified (GPUS=${GPUS_RAW})"
  exit 1
fi

ROOTS_STR="${ROOTS:-results/arc/qwen7b_tuned_v3}"
read -r -a ROOTS_ARR <<< "${ROOTS_STR}"

PROFILES_STR="${PROFILES:-mix111 typo_only synonym_only paraphrase_only typo_stress synonym_stress paraphrase_stress}"
read -r -a PROFILES_ARR <<< "${PROFILES_STR}"

LOG_ROOT="results/arc/_eval_logs_profiles"
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

profile_overrides() {
  # Print overrides (one per line) for a given profile name.
  local name="$1"
  case "${name}" in
    mix111)
      # ID: use checkpoint config (mix111 + budgets). Still isolate caches.
      echo "perturbation.cache_root=cache/perturb_fields_profiles/mix111"
      ;;
    typo_only)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/typo_only"
      echo "perturbation.mix.typo=1.0"
      echo "perturbation.mix.synonym=0.0"
      echo "perturbation.mix.paraphrase=0.0"
      ;;
    synonym_only)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/synonym_only"
      echo "perturbation.mix.typo=0.0"
      echo "perturbation.mix.synonym=1.0"
      echo "perturbation.mix.paraphrase=0.0"
      ;;
    paraphrase_only)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/paraphrase_only"
      echo "perturbation.mix.typo=0.0"
      echo "perturbation.mix.synonym=0.0"
      echo "perturbation.mix.paraphrase=1.0"
      ;;
    typo_stress)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/typo_stress"
      echo "perturbation.mix.typo=1.0"
      echo "perturbation.mix.synonym=0.0"
      echo "perturbation.mix.paraphrase=0.0"
      # Stress: stronger than 'large'
      echo "perturbation.typo_drop_prob=0.15"
      ;;
    synonym_stress)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/synonym_stress"
      echo "perturbation.mix.typo=0.0"
      echo "perturbation.mix.synonym=1.0"
      echo "perturbation.mix.paraphrase=0.0"
      echo "perturbation.replace_ratio=0.35"
      ;;
    paraphrase_stress)
      echo "perturbation.cache_root=cache/perturb_fields_profiles/paraphrase_stress"
      echo "perturbation.mix.typo=0.0"
      echo "perturbation.mix.synonym=0.0"
      echo "perturbation.mix.paraphrase=1.0"
      echo "perturbation.paraphrase_window=14"
      ;;
    *)
      echo "Unknown profile: ${name}" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local ckpt_dir="$2"
  local suite="$3"
  local suite_root="$4"
  local profile="$5"

  local eval_dir="${ckpt_dir}/eval_profiles/${profile}"
  local ckpt_id
  ckpt_id="$(echo -n "${ckpt_dir}::${profile}" | md5sum | awk '{print $1}')"
  local log_file="${LOG_ROOT}/${suite}__${profile}__${ckpt_id}__gpu${gpu}.log"
  mkdir -p "${eval_dir}"

  local rel="${ckpt_dir#${suite_root}/}"
  if [[ "${rel}" == "${ckpt_dir}" ]]; then
    rel="${ckpt_dir##*/}"
  fi

  # Build overrides array.
  local -a OVR
  OVR+=("logging.save_dir=${eval_dir}")
  OVR+=("logging.metrics_path=${eval_dir}/metrics.jsonl")
  OVR+=("logging.final_metrics_path=${eval_dir}/metrics.json")
  OVR+=("logging.matrix_path=${eval_dir}/matrix.json")
  OVR+=("logging.backend=wandb")
  OVR+=("logging.wandb.project=${WANDB_PROJECT}")
  OVR+=("logging.wandb.group=${WANDB_GROUP}")
  OVR+=("logging.wandb.mode=${WANDB_MODE}")
  OVR+=("logging.wandb.name=${suite}/eval_${profile}/${rel}")

  while IFS= read -r line; do
    [[ -n "${line}" ]] && OVR+=("${line}")
  done < <(profile_overrides "${profile}")

  echo "[eval] gpu=${gpu} suite=${suite} profile=${profile} ckpt=${ckpt_dir} log=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m experiments.eval_classification \
    --config "${ckpt_dir}/config_resolved.yaml" \
    --ckpt "${ckpt_dir}" \
    $(printf -- ' --override %q' "${OVR[@]}") \
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

echo "[info] found ${#CKPTS[@]} completed checkpoints"
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  exit 0
fi

# Expand tasks as (ckpt, profile).
TASK_CKPTS=()
TASK_PROFILES=()
for ckpt in "${CKPTS[@]}"; do
  for profile in "${PROFILES_ARR[@]}"; do
    TASK_CKPTS+=("${ckpt}")
    TASK_PROFILES+=("${profile}")
  done
done

echo "[info] total eval tasks: ${#TASK_CKPTS[@]} (ckpt x profiles)"

# One-job-per-GPU scheduler.
declare -A PID_BY_GPU
job_idx=0

for i in "${!TASK_CKPTS[@]}"; do
  ckpt="${TASK_CKPTS[$i]}"
  profile="${TASK_PROFILES[$i]}"

  suite="${ckpt#results/arc/}"
  suite="${suite%%/*}"
  suite_root="results/arc/${suite}"

  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  prev="${PID_BY_GPU[$gpu]:-}"
  if [[ -n "${prev}" ]]; then
    wait "${prev}" || true
    unset PID_BY_GPU["$gpu"]
  fi
  (run_one "${gpu}" "${ckpt}" "${suite}" "${suite_root}" "${profile}") &
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
echo "[done] all profile eval jobs completed."




