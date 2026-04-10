# 实验 Demo 速览（模型 / 数据 / 做法）

## 模型

- **Qwen/Qwen3-VL-2B-Instruct**（同一套 backbone 跑所有任务）
- **4-bit 量化** + **LoRA**（rank 8, α 16，主要改 `q_proj` / `v_proj`）
- 分类/多选：用 **verbalizer**，下一 token 预测标签字母 A/B/C/D（Robot 为 A–F 对应 6 个动作）

## 三个已跑通的 Demo

| Demo | 数据 | 规模（约） | 输入 |
|------|------|------------|------|
| **AG News** | HuggingFace `ag_news`，4 类新闻 | 2k 训 / 500验 / 500 测 | 纯文本 |
| **ARC-Challenge** | `allenai/ai2_arc`子集 ARC-Challenge | 1k 训 / 295 验 / 500 测 | 纯文本多选 |
| **Robot 动作** | 项目内 **合成** 多模态数据（图 + 指令） | 1.5k 训 / 300 验 / 300 测 | 图 + 文 → 6 类离散动作 |

*说明：配置里还有 ScienceQA 多模态多选，与本次「已汇总进 `results_all.csv`」的主线不同；主线是上面三个。*

## 三组对比方法

1. **Base-clean**：只在干净数据上微调  
2. **Base-aug**：干净 + **同样扰动** 的增强数据（无正则项）  
3. **Plugin**：在 Base-aug 相同训练目标上，额外加 **gamma 感知 plugin 正则**（margin 门控 + 转移矩阵谱范数 R_spec，可选 R_stab）

## 扰动（评测与 aug）

- **文本**：拼写扰动、无关句插入、格式改写（AG/ARC）；Robot 还用文本 typo / distractor  
- **多模态（Robot 等）**：模糊、JPEG、缩小再放大；**joint** = 文本 + 图像扰动组合

## 训练流程（简述）

1. 训 Base-clean、Base-aug（各任务、各 seed，如 42 / 123）  
2. 用 **Base-aug** 在 **扰动验证集** 上的真实类 margin 分布定 **γ**（默认分位数 q25）  
3. 用该 γ 训 **Plugin**  
4. **evaluate**：干净 + 各扰动；指标含准确率、**worst-class** 准确率/误差、**VWR_γ**、转移矩阵 **σ_max**

## 跑代码入口

- 全流程编排：`run_all.py` / `run.sh`  
- 结果：`results/results_all.csv`，说明报告：`results/summary.md`
