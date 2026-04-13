#!/bin/bash
set -e

ARDS_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS"
VENV="/LOCAL2/zhuoyun/PAC_robust/ards_venv"
PY="$VENV/bin/python"
MODEL="/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b"
DATA_PATH="$ARDS_DIR/playground/data/eval/scienceqa/llava_train_QCM-A_globalid.json"
IMAGE_FOLDER="$ARDS_DIR/playground/data/eval/scienceqa/images"
OUTPUT_DIR="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/sqa_selection"
export PYTHONPATH="$ARDS_DIR:$PYTHONPATH"
export HF_HOME="/LOCAL2/zhuoyun/hf_cache"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Step A: Collecting representations (3 GPUs)"
echo "============================================"

gpu_list="0,1,2"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY llava/data/get_train_repr.py \
        --model_path "$MODEL" \
        --version v1 \
        --data_path "$DATA_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --save_prefix 'weighted_attn' \
        --selection_strategy "weighted_attn" \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

echo "Waiting for all representation extraction to finish..."
wait
echo "Step A done."

echo "============================================"
echo "Step A2: Merging representations"
echo "============================================"

$PY llava/data/merge_repr_grad_files.py \
    --output_dir "$OUTPUT_DIR/reps/weighted_attn" \
    --prefix reps \
    --woproj \
    --save_normalize

echo "Merge done."
echo "============================================"
echo "Step B: Clustering subgroups"
echo "============================================"

$PY -c "
import sys
sys.path.insert(0, '$ARDS_DIR')
from llava.data.cluster_subgroup import main
main(
    vector_path='$OUTPUT_DIR/reps/weighted_attn/all_orig.pt',
    dataset_file='$DATA_PATH',
    n_components=10,
    n=200,
    niter=20
)
print('Clustering done.')
"

echo "============================================"
echo "All steps complete."
echo "============================================"
