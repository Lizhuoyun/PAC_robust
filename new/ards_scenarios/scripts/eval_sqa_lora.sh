#!/bin/bash
set -e

ARDS_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS"
VENV="/LOCAL2/zhuoyun/PAC_robust/ards_venv"
PY="$VENV/bin/python"
SCENARIOS="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"

export PYTHONPATH="$ARDS_DIR:$PYTHONPATH"
export HF_HOME="/LOCAL2/zhuoyun/hf_cache"

CKPT="${1:?Usage: $0 <checkpoint_dir> [gpu_id]}"
GPU="${2:-0}"
BASE_MODEL="/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b"
CKPT_NAME=$(basename "$CKPT")

cd "$ARDS_DIR"

echo "============================================"
echo "Evaluating: $CKPT_NAME"
echo "============================================"

EVAL_DIR="$SCENARIOS/eval_outputs/$CKPT_NAME"
mkdir -p "$EVAL_DIR"

echo "--- Clean Evaluation ---"
CUDA_VISIBLE_DEVICES=$GPU $PY llava/eval/model_vqa_science.py \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    --question-file playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder playground/data/eval/scienceqa/images \
    --answers-file "$EVAL_DIR/clean.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0

echo "--- Symbol Attack (SA) Evaluation ---"
CUDA_VISIBLE_DEVICES=$GPU $PY llava/eval/model_vqa_science.py \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    --question-file playground/data/eval/scienceqa/llava_test_CQM-A_convertedABCDE-QWERT.json \
    --image-folder playground/data/eval/scienceqa/images \
    --answers-file "$EVAL_DIR/sa.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0

echo "--- Position Attack (PA) Evaluation ---"
CUDA_VISIBLE_DEVICES=$GPU $PY llava/eval/model_vqa_science_option_attack.py \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    --question-file playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder playground/data/eval/scienceqa/images \
    --answers-file "$EVAL_DIR/pa.jsonl" \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --eval-img

echo "============================================"
echo "All evaluations saved to: $EVAL_DIR"
echo "============================================"
