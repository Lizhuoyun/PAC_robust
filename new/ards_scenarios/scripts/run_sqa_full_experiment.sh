#!/bin/bash
set -e

ARDS_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS"
VENV="/LOCAL2/zhuoyun/PAC_robust/ards_venv"
PY="$VENV/bin/python"
SCENARIOS="/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"
CKPT_DIR="$SCENARIOS/checkpoints"

export PYTHONPATH="$ARDS_DIR:$SCENARIOS:$PYTHONPATH"
export HF_HOME="/LOCAL2/zhuoyun/hf_cache"

DATA_PATH="$SCENARIOS/sqa_selection/scienceqa_selected_subset.json"

mkdir -p "$CKPT_DIR" "$SCENARIOS/logs" "$SCENARIOS/results"

echo "============================================"
echo " ScienceQA Full Experiment Pipeline"
echo " Phase 1: LoRA baseline (GPU 0)"
echo " Phase 2: Gamma calibration (GPU 0)"
echo " Phase 3: Plugin-LoRA q10/q25/q50 (GPU 0,1,2)"
echo " Phase 4: Unified evaluation (GPU 0,1,2)"
echo "============================================"

###############################################################################
# Phase 1: Train LoRA baseline
###############################################################################
echo ""
echo ">>> Phase 1: Training LoRA baseline on GPU 0..."

LORA_CKPT="$CKPT_DIR/sqa_lora"
if [ -f "$LORA_CKPT/adapter_config.json" ]; then
    echo "  LoRA checkpoint already exists, skipping training."
else
    cd "$ARDS_DIR"
    CUDA_VISIBLE_DEVICES=0 $PY llava/train/train_mem.py \
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
        --output_dir "$LORA_CKPT" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none \
        2>&1 | tee "$SCENARIOS/logs/train_lora.log"
    echo "  LoRA training complete."
fi

###############################################################################
# Phase 2: Gamma calibration
###############################################################################
echo ""
echo ">>> Phase 2: Gamma calibration from LoRA checkpoint..."

GAMMA_FILE="$SCENARIOS/gamma_calibration.json"
if [ -f "$GAMMA_FILE" ]; then
    echo "  Gamma calibration already exists, skipping."
else
    cd "$SCENARIOS"
    CUDA_VISIBLE_DEVICES=0 $PY sqa_gamma_calibrate.py \
        --ckpt_dir "$LORA_CKPT" \
        --data_path "$DATA_PATH" \
        --gpu 0 \
        --n_samples 500 \
        --output "$GAMMA_FILE" \
        2>&1 | tee "$SCENARIOS/logs/gamma_calibrate.log"
    echo "  Gamma calibration complete."
fi

cat "$GAMMA_FILE"

GAMMA_Q10=$(python3 -c "import json; d=json.load(open('$GAMMA_FILE')); print(d['q10'])")
GAMMA_Q25=$(python3 -c "import json; d=json.load(open('$GAMMA_FILE')); print(d['q25'])")
GAMMA_Q50=$(python3 -c "import json; d=json.load(open('$GAMMA_FILE')); print(d['q50'])")

echo ""
echo "  Gamma values: q10=$GAMMA_Q10, q25=$GAMMA_Q25, q50=$GAMMA_Q50"

###############################################################################
# Phase 3: Plugin-LoRA training (3 GPUs parallel)
###############################################################################
echo ""
echo ">>> Phase 3: Training Plugin-LoRA on 3 GPUs in parallel..."

train_plugin() {
    local QNAME=$1
    local GAMMA_VAL=$2
    local GPU_ID=$3

    local OUT_DIR="$CKPT_DIR/sqa_plugin_${QNAME}"

    if [ -f "$OUT_DIR/adapter_config.json" ]; then
        echo "  Plugin-$QNAME checkpoint exists, skipping."
        return 0
    fi

    echo "  Starting Plugin-$QNAME (gamma=$GAMMA_VAL) on GPU $GPU_ID..."
    cd "$ARDS_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PY llava/train/train_plugin.py \
        --gamma "$GAMMA_VAL" --plugin_alpha 0.1 --plugin_kappa 0.5 --plugin_beta 0.0 \
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
        --output_dir "$OUT_DIR" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none \
        2>&1 | tee "$SCENARIOS/logs/train_plugin_${QNAME}.log"
    echo "  Plugin-$QNAME training complete."
}

train_plugin "q10" "$GAMMA_Q10" 0 &
PID_Q10=$!
train_plugin "q25" "$GAMMA_Q25" 1 &
PID_Q25=$!
train_plugin "q50" "$GAMMA_Q50" 2 &
PID_Q50=$!

echo "  Waiting for all plugin training jobs..."
wait $PID_Q10 $PID_Q25 $PID_Q50
echo "  All plugin training complete."

###############################################################################
# Phase 4: Unified evaluation (3 GPUs parallel)
###############################################################################
echo ""
echo ">>> Phase 4: Evaluating all checkpoints..."

eval_checkpoint() {
    local METHOD=$1
    local CKPT=$2
    local GAMMA_VAL=$3
    local GPU_ID=$4

    echo "  Evaluating $METHOD (gamma=$GAMMA_VAL) on GPU $GPU_ID..."
    cd "$SCENARIOS"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PY sqa_unified_eval.py \
        --ckpt_dir "$CKPT" \
        --gamma "$GAMMA_VAL" \
        --gpu 0 \
        --eval_types "clean,sa,pa" \
        --output "$SCENARIOS/results/eval_${METHOD}.json" \
        2>&1 | tee "$SCENARIOS/logs/eval_${METHOD}.log"
}

eval_checkpoint "lora" "$LORA_CKPT" "$GAMMA_Q25" 0 &
PID_E1=$!
eval_checkpoint "plugin_q10" "$CKPT_DIR/sqa_plugin_q10" "$GAMMA_Q10" 1 &
PID_E2=$!
eval_checkpoint "plugin_q25" "$CKPT_DIR/sqa_plugin_q25" "$GAMMA_Q25" 2 &
PID_E3=$!

wait $PID_E1 $PID_E2 $PID_E3

eval_checkpoint "plugin_q50" "$CKPT_DIR/sqa_plugin_q50" "$GAMMA_Q50" 0

echo ""
echo ">>> All evaluations complete."

###############################################################################
# Phase 5: Aggregate results
###############################################################################
echo ""
echo ">>> Phase 5: Aggregating results..."

cd "$SCENARIOS"
$PY sqa_aggregate_results.py

echo ""
echo "============================================"
echo " Full experiment pipeline complete!"
echo " Results: $SCENARIOS/results/"
echo "============================================"
