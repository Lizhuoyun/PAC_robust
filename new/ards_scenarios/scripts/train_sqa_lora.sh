#!/bin/bash
set -e

ARDS_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS"
VENV="/LOCAL2/zhuoyun/PAC_robust/ards_venv"
PY="$VENV/bin/python"
SCENARIOS="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"

export PYTHONPATH="$ARDS_DIR:$PYTHONPATH"
export HF_HOME="/LOCAL2/zhuoyun/hf_cache"

DATA_VARIANT="${1:-selected30}"
GPU="${2:-0}"

case "$DATA_VARIANT" in
    selected30)
        DATA_PATH="$SCENARIOS/sqa_selection/scienceqa_selected_subset_top30.json"
        SUFFIX="lora_selected30"
        ;;
    selected50)
        DATA_PATH="$SCENARIOS/sqa_selection/scienceqa_selected_subset_top50.json"
        SUFFIX="lora_selected50"
        ;;
    selected70)
        DATA_PATH="$SCENARIOS/sqa_selection/scienceqa_selected_subset_top70.json"
        SUFFIX="lora_selected70"
        ;;
    full)
        DATA_PATH="$ARDS_DIR/playground/data/eval/scienceqa/llava_train_QCM-A_globalid.json"
        SUFFIX="lora_full"
        ;;
    *)
        echo "Usage: $0 [selected30|selected50|selected70|full] [gpu_id]"
        exit 1
        ;;
esac

OUTPUT_DIR="$SCENARIOS/checkpoints/sqa_${SUFFIX}"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Training ScienceQA LoRA: $DATA_VARIANT"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "============================================"

cd "$ARDS_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PY llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path /LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$ARDS_DIR/playground/data/eval/scienceqa/images" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "============================================"
echo "Training complete: $OUTPUT_DIR"
echo "============================================"
