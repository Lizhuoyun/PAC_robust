# ScienceQA LoRA / Plugin-LoRA 实验准备文档

## 1. 数据底座

### Selected Subset（ARDS-style selection 生成）

| 文件 | 样本数 | 描述 |
|------|--------|------|
| `sqa_selection/scienceqa_selected_subset.json` | 3817 | 默认 subset (top 30%) |
| `sqa_selection/scienceqa_selected_subset_top30.json` | 3817 | top 30% |
| `sqa_selection/scienceqa_selected_subset_top50.json` | 6363 | top 50% |
| `sqa_selection/scienceqa_selected_subset_top70.json` | 8908 | top 70% |
| `sqa_selection/scienceqa_selected_scores.json` | 12226 | 所有训练样本的 influence scores |

### 训练数据格式

每条数据包含: id, global_id, image (路径), conversations (human/gpt 对话)

### 图片路径

- Image folder: `/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/images`
- 训练集图片 6218 张全部在位

## 2. 普通 LoRA 微调入口

### 训练脚本

入口: `llava/train/train_mem.py` -> `llava/train/train.py: train()`
参考: `scripts/v1_5/finetune_task_lora.sh`

### 可直接使用的命令（单 GPU）

```
cd /LOCAL2/zhuoyun/PAC_robust/ARDS
export PYTHONPATH="/LOCAL2/zhuoyun/PAC_robust/ARDS:$PYTHONPATH"
export HF_HOME="/LOCAL2/zhuoyun/hf_cache"
PY="/LOCAL2/zhuoyun/PAC_robust/ards_venv/bin/python"

CUDA_VISIBLE_DEVICES=0 $PY llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path /LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b \
    --version v1 \
    --data_path /LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/sqa_selection/scienceqa_selected_subset.json \
    --image_folder /LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/checkpoints/sqa_lora_selected30 \
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
    --lazy_preprocess True
```

## 3. Plugin-LoRA 插入点

### 代码位置

| 模块 | 文件 | 说明 |
|------|------|------|
| 训练入口 | `llava/train/train.py: train()` | 模型加载、数据构建、启动训练 |
| Trainer | `llava/train/llava_trainer.py: LLaVATrainer` | 继承自 HuggingFace Trainer |
| 模型 | `llava/model/language_model/llava_llama.py` | LLaVA 模型定义 |

### Plugin 插入策略（推荐方案 A：Trainer 层面）

在 `LLaVATrainer` 中添加 `compute_loss()`:

```python
class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss  # cross-entropy
        # === Plugin-LoRA 插入点 ===
        # 1. 提取 verbalizer logits (ABCDE 对应 token)
        # 2. 计算 margin: max_wrong - correct logit
        # 3. gamma-aware gate
        # 4. transition matrix regularization
        # 5. spectral regularizer on LoRA weights
        # loss = loss + gamma * plugin_loss
        if return_outputs:
            return loss, outputs
        return loss
```

优点：不需修改模型代码，可以访问完整 outputs，易于开关调试。

## 4. 评测入口

### Clean 评测
```
$PY llava/eval/model_vqa_science.py \
    --model-path <checkpoint> --model-base /LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b \
    --question-file playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder playground/data/eval/scienceqa/images \
    --answers-file <output>.jsonl --conv-mode vicuna_v1 --temperature 0
```

### SA (Symbol Attack) 评测
同上但 `--question-file` 换为 `llava_test_CQM-A_convertedABCDE-QWERT.json`

### PA (Position Attack) 评测
用 `model_vqa_science_option_attack.py`，加 `--eval-img` 参数

**注意**: LoRA 模型评测需要 `--model-base` 指向基础模型。

## 5. 实验矩阵

### Phase 1: 基础对比
| 实验 | 数据 | 方法 | 评测 |
|------|------|------|------|
| Pre-trained baseline | - | 零射 | Clean 70.22%, SA 49.61%, PA 46.26% |
| LoRA on selected-30% | 3817 条 | LoRA r=128 | Clean/SA/PA |
| LoRA on full-data | 12726 条 | LoRA r=128 | Clean/SA/PA |

### Phase 2: Plugin-LoRA
| gamma | 数据 | 方法 |
|-------|------|------|
| 0.01, 0.1, 0.5, 1.0, 2.0, 5.0 | selected-30% | Plugin-LoRA |

## 6. 关键路径
```
数据: sqa_selection/scienceqa_selected_subset.json (3817条)
图片: ARDS/playground/data/eval/scienceqa/images/
模型: /LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b
训练: llava/train/train_mem.py -> LLaVATrainer
Plugin: LLaVATrainer.compute_loss()
环境: /LOCAL2/zhuoyun/PAC_robust/ards_venv
```
