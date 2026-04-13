# ARDS 仓库可复用性审计报告

**审计时间**: 2026-04-10  
**仓库路径**: `/LOCAL2/zhuoyun/PAC_robust/ARDS`  
**论文**: "Data Selection Matters: Towards Robust Instruction Tuning of Large Multimodal Models" (NeurIPS 2025)

---

## 一、仓库总体概述

ARDS 是基于 **LLaVA-v1.5** 架构的多模态鲁棒数据选择框架，分三步：

```
[warm-up LoRA checkpoint]
        ↓
[get_train_repr.py] → 提取训练集表征/梯度
        ↓
[build_worst_subgroup.py + cluster_subgroup.py] → 构建最差评测子组（视觉/文本扰动）
        ↓
[matching_worst_subgroup.py] → 余弦相似度选出接近最差子组的训练样本
        ↓
[finetune_7b.sh / finetune_lora.sh] → 在选出数据上训练
        ↓
[eval/sqa.sh 等] → clean + PA + SA 评测
```

---

## 二、支持任务清单

| 任务 | 评测方式 | 攻击评测 | 脚本完整性 | 当前数据状态 |
|------|----------|---------|-----------|------------|
| **ScienceQA** | 选择题 ABCDE | ✅ PA + SA | ✅ 完整 | 注释文件 ✓，**缺图像** |
| **GQA** | 开放答案 | ✅ PA | ✅ 完整 | **全部需下载** |
| **GQA-OOD** | 开放答案 | ✅ PA | ✅ + create_llava_file.py | **全部需下载** |
| **TextVQA** | 开放答案 | ⚠️ 无显式攻击脚本 | ✅ 完整 | **全部需下载** |
| **SEED-Bench** | 选择题 ABCD | ✅ PA (ABCD→QWER) | ✅ 完整 | **全部需下载** |
| **MMBench** | 选择题 | ❌ | ❌ **BROKEN** (缺 model_vqa_mmbench.py) | **全部需下载** |

---

## 三、脚本完整索引

### 3.1 数据选择
| 脚本 | 功能 |
|------|------|
| `scripts/v1_5/selection.sh` | 选择总入口 |
| `llava/data/get_train_repr.py` | 提取训练集 weighted_attn 表征（需 warm-up LoRA） |
| `llava/data/merge_repr_grad_files.py` | 合并多 GPU 表征文件 |
| `llava/data/build_worst_subgroup.py` | 构建最差子组（视觉/文本扰动） |
| `llava/data/cluster_subgroup.py` | FAISS 聚类（需 faiss-gpu） |
| `llava/data/matching_worst_subgroup.py` | 余弦相似度选择（ARDS 核心） |

### 3.2 训练脚本
| 脚本 | 模型 | 模式 |
|------|------|------|
| `scripts/v1_5/finetune_7b.sh` | vicuna-7b-v1.5 | 全量（从头建 LLaVA-7b） |
| `scripts/v1_5/finetune.sh` | LLaVA-v1.5-13b | 全量 |
| `scripts/v1_5/finetune_lora.sh` | LLaVA-v1.5-13b | LoRA (r=128, α=256) |
| `scripts/v1_5/finetune_task_lora.sh` | LLaVA-v1.5-13b | Task-specific LoRA |
| 训练入口 | `llava/train/train_mem.py` | DeepSpeed 入口 |
| Trainer | `llava/train/llava_trainer.py` | 基于 HF Trainer |

### 3.3 评测脚本
| 脚本 | 任务 | 包含攻击 |
|------|------|---------|
| `scripts/v1_5/eval/sqa.sh` | ScienceQA | ✅ clean + PA + SA |
| `scripts/v1_5/eval/gqa.sh` | GQA | ✅ multi-GPU |
| `scripts/v1_5/eval/seed.sh` | SEED-Bench | ✅ PA |
| `scripts/v1_5/eval/textvqa.sh` | TextVQA | 只 clean |
| `scripts/v1_5/eval/mmbench.sh` | MMBench | ❌ 脚本缺失 |
| `llava/eval/model_vqa_science.py` | ScienceQA 推理 | |
| `llava/eval/model_vqa_science_option_attack.py` | ScienceQA 攻击 | PA/SA |
| `llava/eval/model_vqa_loader.py` | 通用推理 | |
| `llava/eval/model_vqa_loader_option_attack.py` | 通用攻击推理 | |
| `llava/eval/eval_science_qa.py` | ScienceQA 指标 | |
| `llava/eval/eval_textvqa.py` | TextVQA 指标 | |

---

## 四、模型权重

| 模型 | 用途 | 大小 |
|------|------|------|
| `liuhaotian/llava-v1.5-7b` | **Full-data 全量 baseline**（即论文 baseline） | ~14GB |
| `liuhaotian/llava-v1.5-13b` | 13B baseline | ~26GB |
| `lmsys/vicuna-7b-v1.5` | 训练用 base LLM | ~14GB |
| `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` | 预训练 connector | ~1GB |
| `openai/clip-vit-large-patch14-336` | Vision encoder | ~1.7GB |
| ARDS LoRA checkpoint | Google Drive（论文提供链接） | ~500MB |

---

## 五、任务可运行性评估

| 任务 | 状态 | 缺少什么 | 工作量 |
|------|------|---------|-------|
| **ScienceQA eval** | ⭐⭐⭐ 几乎就绪 | 测试图像 + LLaVA-v1.5-7b 模型 | 低 |
| **GQA eval** | ⭐⭐ 需下载 | 问题 JSON + 图像 (~20GB) | 中 |
| **GQA-OOD eval** | ⭐⭐ 需下载 | OOD 问题 JSON + GQA 图像 | 中 |
| **TextVQA eval** | ⭐⭐ 需下载 | 注释 + 图像 (~8GB) | 中 |
| **SEED-Bench eval** | ⭐ 较重 | 完整数据含视频 (~100GB+) | 高 |
| **MMBench eval** | ❌ 需修复 | model_vqa_mmbench.py 缺失 | 需补文件 |
| **全量训练 baseline** | ⭐ 很重 | mix665k 数据 + 所有图像 | 极高 |
| **LoRA 训练** | ⭐⭐ 中等 | 同上，但显存需求低 | 高 |

---

## 六、Selection / Training / Eval 各模块位置

| 阶段 | 路径 |
|------|------|
| **选择入口** | `scripts/v1_5/selection.sh` |
| **选择核心** | `llava/data/matching_worst_subgroup.py` |
| **已选数据输出** | `LLaVA_output/llava_7b-v1.5-warmup/reps/weighted_attn/` |
| **训练入口** | `llava/train/train_mem.py`（deepspeed） |
| **Trainer 类** | `llava/train/llava_trainer.py` |
| **模型构建** | `llava/model/builder.py` → `load_pretrained_model()` |
| **模型类** | `llava/model/language_model/llava_llama.py` |
| **clean 评测** | `scripts/v1_5/eval/*.sh` |
| **PA/SA 评测** | 同上 sh 文件中的 option_attack 部分 |

---

## 七、环境依赖冲突

| 依赖 | ARDS 要求 | 现有 venv | 状态 |
|------|----------|---------|------|
| transformers | **4.37.2** | **4.57.5** | ❌ 不兼容 |
| peft | **0.13.1** | **0.5.0** | ❌ 不兼容 |
| deepspeed | 0.12.6 | 未安装 | ❌ |
| flash-attn | latest | 未安装 | ❌ |
| faiss-gpu | 1.8.0 | 未安装 | ❌ |

**解决方案**: 已创建专用 conda 环境 `ards` (Python 3.10)。

---

## 八、最适合作为 Plugin 对比底座的任务

1. **ScienceQA** ⭐⭐⭐ - 选择题 → 天然 discrete output，PA+SA 完整，verbalizer = A/B/C/D/E
2. **SEED-Bench** ⭐⭐ - 选择题 ABCD，12 个能力维度
3. **GQA-OOD** ⭐⭐ - 开放答案 worst-case 鲁棒性
4. **TextVQA** ⭐ - 补充 OCR 维度
