# 后续接入普通 LoRA / Plugin-LoRA 的说明

**创建时间**: 2026-04-10  
**当前状态**: baseline 搭建阶段，plugin 尚未实现

---

## 一、当前训练入口

### 主入口
```
llava/train/train_mem.py   ← DeepSpeed 推荐入口（省显存）
llava/train/train.py       ← 标准入口（无 DeepSpeed 也可用）
```

启动方式（以 7b LoRA 为例）：
```bash
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    ...
```

### Trainer 类
```
llava/train/llava_trainer.py: LLaVATrainer
```
- 继承自 `transformers.Trainer`
- 覆写了 `_get_train_dataloader`（modality-length 分组采样）
- 覆写了 `_save_checkpoint`（分离保存 mm_projector）
- **未覆写 `compute_loss`**（使用 HF 默认 LM loss）

---

## 二、模型构建位置

```python
# llava/model/builder.py
load_pretrained_model(model_path, model_base, model_name, ...)
    → 返回 (tokenizer, model, image_processor, context_len)
```

模型类层级：
```
LlavaLlamaForCausalLM          ← llava/model/language_model/llava_llama.py
├── LlavaMetaModel (Mixin)      ← llava/model/llava_arch.py
│   ├── vision_tower            ← CLIPVisionModel
│   └── mm_projector            ← MLP2x (mlp2x_gelu)
└── LlamaForCausalLM            ← transformers
    └── LlamaModel
        └── LlamaDecoderLayer × N
            ├── LlamaAttention  ← LoRA 的最佳注入位置
            └── LlamaMLP
```

LoRA 通过 `peft` 注入，由 `--lora_enable True` 控制：
```python
# llava/train/train.py: train() 函数
if training_args.lora_enable:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, ...)
    model = get_peft_model(model, lora_config)
```

---

## 三、最适合加普通 LoRA 的位置

**方式 1：直接用现有 finetune_task_lora.sh（推荐）**

已有脚本 `scripts/v1_5/finetune_task_lora.sh` 完整支持 LoRA 微调：
```bash
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    ...
```

修改 `--data_path` 切换数据集（full-data / selected-data）即可。

**方式 2：扫描不同 LoRA r/α**

在 `ards_scenarios/fixes/` 下写参数扫描脚本，保持训练逻辑不动。

**LoRA 目标模块**（建议）：
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

---

## 四、Plugin-LoRA 最适合插入的代码位置

### 4.1 Verbalizer Logits 提取

```python
# 位置: llava/train/llava_trainer.py
# 在 compute_loss() 或 prediction_step() 中

# 目前 compute_loss 未覆写，使用 HF 默认
# 插入点: 覆写 LLaVATrainer.compute_loss()

def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    logits = outputs.logits  # (B, T, vocab_size)
    
    # Plugin: 从 logits 提取 verbalizer 位置的 log-probs
    # verbalizer_token_ids = tokenizer.convert_tokens_to_ids(["A","B","C","D","E"])
    # option_logits = logits[:, answer_position, verbalizer_token_ids]  # (B, num_options)
```

### 4.2 Margin 计算

```python
# 在 compute_loss() 中，得到 option_logits 后：
# margin = option_logits[correct_option] - max(option_logits[wrong_options])
# 这是 Plugin 的核心统计量
```

### 4.3 Gamma-aware Gate

```python
# Plugin module: 在 llava/model/ 下新建 plugin_lora.py
# 在 LlavaLlamaForCausalLM.forward() 的残差连接处注入 gate
```

### 4.4 Transition Matrix

```python
# 位置: 在 compute_loss() 中的 label_smoothing / loss 计算之前
# transition_matrix: (num_options, num_options) → 将 option logits 重映射
# 最适合在 verbalizer logits 提取之后、loss 计算之前注入
```

### 4.5 Spectral Regularizer

```python
# 位置: compute_loss() 返回前，加到 loss 上
# loss = ce_loss + gamma * spectral_reg
```

---

## 五、哪些任务最适合第一批 Plugin 对比实验

### 推荐顺序

**1. ScienceQA（强烈推荐作为第一个）**
- 理由：
  - 选择题 ABCDE = 5 个明确 verbalizer tokens，margin 计算天然
  - 有 PA + SA 两种攻击，能直观看到 plugin 的鲁棒性提升
  - 数据量适中（train ~12k），实验快
  - 已有 baseline eval 脚本，改动最小
  
**2. SEED-Bench（第二批）**
- 选择题 ABCD = 4 个 verbalizer
- 12 个能力维度，可分析 plugin 对不同维度的影响
  
**3. GQA-OOD（开放答案对比）**
- 验证 plugin 在非选择题场景的泛化性
- OOD 分布提供天然的 worst-case 评测

---

## 六、普通 LoRA vs Plugin-LoRA 对比实验设计建议

### 实验矩阵

| 模型变体 | 数据 | 任务 | 评测 |
|---------|------|------|------|
| llava-v1.5-7b (pretrained) | - | SQA, GQA | clean + PA + SA |
| full-data LoRA | mix665k | SQA, GQA | clean + PA + SA |
| ARDS-selected LoRA | ~5% mix665k | SQA, GQA | clean + PA + SA |
| full-data Plugin-LoRA (γ=0.1) | mix665k | SQA, GQA | clean + PA + SA |
| full-data Plugin-LoRA (γ=0.5) | mix665k | SQA, GQA | clean + PA + SA |
| ARDS-selected Plugin-LoRA (γ=0.1) | ~5% mix665k | SQA, GQA | clean + PA + SA |

### 关键超参扫描

- LoRA r: {64, 128} 
- LoRA α: {128, 256}
- Plugin γ: {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}（这是你的核心创新参数）

---

## 七、最可能复用哪套 baseline

**普通 LoRA 微调最直接复用**:
```
scripts/v1_5/finetune_task_lora.sh
```
只需修改 `--data_path` 和 `--output_dir`。

**Plugin-LoRA 最适合的插入路径**:
```
llava/train/llava_trainer.py: LLaVATrainer.compute_loss() (覆写)
llava/model/language_model/llava_llama.py: forward() (加 gate)
```
两处改动即可，不影响原有训练逻辑。

---

## 八、不应该动的地方

- `llava/model/builder.py` - 模型加载逻辑，保持原样
- `llava/data/` - 数据选择流程，保持原样
- `llava/eval/` - 评测脚本，保持原样
- DeepSpeed config (`scripts/zero3.json`) - 保持原样
