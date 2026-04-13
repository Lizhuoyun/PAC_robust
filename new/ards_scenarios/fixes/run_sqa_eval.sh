#!/bin/bash
# run_sqa_eval.sh
# 跑通 ScienceQA 完整评测闭环（clean + PA + SA）
# 使用 liuhaotian/llava-v1.5-7b 作为 full-data baseline
#
# 使用方法:
#   CUDA_VISIBLE_DEVICES=0,1,2 bash fixes/run_sqa_eval.sh
#   或
#   CUDA_VISIBLE_DEVICES=0 bash fixes/run_sqa_eval.sh [MODEL_PATH]

set -e

MODEL_PATH="${1:-liuhaotian/llava-v1.5-7b}"
ARDS_ROOT="/LOCAL2/zhuoyun/PAC_robust/ARDS"
OUTPUT_DIR="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs/scienceqa"
SQA_EVAL_DIR="${ARDS_ROOT}/playground/data/eval/scienceqa"
CKPT_NAME="$(basename $MODEL_PATH)"

mkdir -p "${OUTPUT_DIR}/clean"
mkdir -p "${OUTPUT_DIR}/pa_attack"
mkdir -p "${OUTPUT_DIR}/sa_attack"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ards

cd "${ARDS_ROOT}"

export HF_HOME=/LOCAL2/zhuoyun/hf_cache
export TRANSFORMERS_CACHE=/LOCAL2/zhuoyun/hf_cache

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"

# ==========================================
# Step 1: Clean Evaluation
# ==========================================
echo ""
echo "=== Step 1: Clean Evaluation ==="
CLEAN_ANSWERS="${OUTPUT_DIR}/clean/${CKPT_NAME}.jsonl"

python -m llava.eval.model_vqa_science \
    --model-path "${MODEL_PATH}" \
    --question-file "${SQA_EVAL_DIR}/llava_test_CQM-A.json" \
    --image-folder "${SQA_EVAL_DIR}/images" \
    --answers-file "${CLEAN_ANSWERS}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir "${SQA_EVAL_DIR}" \
    --result-file "${CLEAN_ANSWERS}" \
    --output-file "${OUTPUT_DIR}/clean/${CKPT_NAME}_output.jsonl" \
    --output-result "${OUTPUT_DIR}/clean/${CKPT_NAME}_result.json"

echo "Clean eval done. Results in ${OUTPUT_DIR}/clean/"

# ==========================================
# Step 2: PA (Position Attack) - ABCDE options
# ==========================================
echo ""
echo "=== Step 2: PA Attack (ABCDE permutation) ==="
PA_ANSWERS="${OUTPUT_DIR}/pa_attack/${CKPT_NAME}.jsonl"

python -m llava.eval.model_vqa_science_option_attack \
    --model-path "${MODEL_PATH}" \
    --question-file "${SQA_EVAL_DIR}/llava_test_CQM-A.json" \
    --image-folder "${SQA_EVAL_DIR}/images" \
    --answers-file "${PA_ANSWERS}" \
    --single-pred-prompt \
    --temperature 0 \
    --options "A" "B" "C" "D" "E" \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir "${SQA_EVAL_DIR}" \
    --result-file "${PA_ANSWERS}" \
    --output-file "${OUTPUT_DIR}/pa_attack/${CKPT_NAME}_output.jsonl" \
    --output-result "${OUTPUT_DIR}/pa_attack/${CKPT_NAME}_result.json"

echo "PA eval done. Results in ${OUTPUT_DIR}/pa_attack/"

# ==========================================
# Step 3: SA (Symbol Attack) - QWERT options
# ==========================================
echo ""
echo "=== Step 3: SA Attack (QWERT symbols) ==="
SA_ANSWERS="${OUTPUT_DIR}/sa_attack/${CKPT_NAME}.jsonl"

python -m llava.eval.model_vqa_science_option_attack \
    --model-path "${MODEL_PATH}" \
    --question-file "${SQA_EVAL_DIR}/llava_test_CQM-A.json" \
    --image-folder "${SQA_EVAL_DIR}/images" \
    --answers-file "${SA_ANSWERS}" \
    --single-pred-prompt \
    --temperature 0 \
    --options "Q" "W" "E" "R" "T" \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir "${SQA_EVAL_DIR}" \
    --result-file "${SA_ANSWERS}" \
    --output-file "${OUTPUT_DIR}/sa_attack/${CKPT_NAME}_output.jsonl" \
    --output-result "${OUTPUT_DIR}/sa_attack/${CKPT_NAME}_result.json"

echo "SA eval done. Results in ${OUTPUT_DIR}/sa_attack/"

# ==========================================
# Step 4: Summary
# ==========================================
echo ""
echo "=== Summary ==="
python3 << PYEOF
import json, os

def read_result(path):
    if os.path.exists(path):
        d = json.load(open(path))
        return d.get('acc', 'N/A')
    return 'missing'

base = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs/scienceqa"
ckpt = "${CKPT_NAME}"

clean = read_result(f"{base}/clean/{ckpt}_result.json")
pa    = read_result(f"{base}/pa_attack/{ckpt}_result.json")
sa    = read_result(f"{base}/sa_attack/{ckpt}_result.json")

print(f"Model: {ckpt}")
print(f"  Clean accuracy: {clean:.2f}%" if isinstance(clean, float) else f"  Clean accuracy: {clean}")
print(f"  PA accuracy:    {pa:.2f}%"    if isinstance(pa, float)    else f"  PA accuracy:    {pa}")
print(f"  SA accuracy:    {sa:.2f}%"    if isinstance(sa, float)    else f"  SA accuracy:    {sa}")
PYEOF

echo "=== ScienceQA eval complete ==="
