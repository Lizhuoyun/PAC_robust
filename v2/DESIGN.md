# 实验设计文档：Language Model Robustness + Gamma-Aware Plugin Regularizer

---

## 一、实验目标

验证 gamma-aware plugin regularizer 能否：
1. 在不损伤 clean performance 前提下提高扰动鲁棒性
2. 改善 worst-class fragile behavior
3. 降低 VWR_gamma 和 transition matrix spectral norm（σ_max）
4. 分析 gamma 取值对 clean–robustness trade-off 的影响

---

## 二、模型选择

**选用：Qwen/Qwen3-VL-2B-Instruct（文本模式，忽略图像输入）**

理由：
- 本地 `/LOCAL2/zhuoyun/hf_cache/hub/` 中唯一可用的小模型
- 2B 参数，配合 4-bit + LoRA 可在单卡 24GB 内完成训练
- 已在旧版本验证了 tokenizer / verbalizer 兼容性
- 本实验纯文本任务无需视觉能力，VLM 与纯文本 LM 在文本推理上性能相当

配置：
- 4-bit NF4 量化（BitsAndBytes）
- LoRA：rank=16，alpha=32，target_modules = [q_proj, v_proj, k_proj, o_proj]
- 训练精度：bfloat16

---

## 三、任务与数据集

**选用：ARC-Challenge（4 类多选推理）**

理由：
- 标准多选格式，verbalizer 设计最清晰（A/B/C/D）
- 难度适中：clean acc 约 80%，有改善空间
- 类别均衡，worst-class 分析有意义
- 已有本地缓存（`allenai/ai2_arc`）

数据规模（balanced subsample）：
- Train：1 000
- Val：295（原验证集全量）
- Test：500

---

## 四、方法比较组

### A. 基础方法

| ID | 方法名 | 训练目标 |
|----|--------|----------|
| 1 | **Base-clean** | CE(clean) |
| 2 | **Base-aug** | (CE(clean) + CE(text_pert)) / 2 |
| 3 | **Base-aug + Plugin** | Base-aug 目标 + α·R_spec + β·R_stab |

### B. 嵌入空间正则方法

| ID | 方法名 | 训练目标 |
|----|--------|----------|
| 4 | **R3F** | CE(clean) + λ·KL(p_clean ‖ p_emb_noise) |
| 5 | **R3F + Plugin** | R3F 目标 + plugin(emb_noise 的 logits) |
| 6 | **SMART** | CE(clean) + λ·KL(p_clean ‖ p_adv_emb) |
| 7 | **SMART + Plugin** | SMART 目标 + plugin(adv_emb 的 logits) |

### C. 权重空间正则方法

| ID | 方法名 | 训练目标 |
|----|--------|----------|
| 8 | **AWP** | CE(w) + CE(w + δw_adv)，δw 只扰 LoRA 参数 |
| 9 | **AWP + Plugin** | AWP 目标 + plugin(w_adv 的 logits) |

**说明**：
- Plugin 为可插拔模块，作用于"某种扰动下的 logits"
- +Plugin 变体中的"扰动 logits"来源与各基础方法的扰动机制一致
- 这样每个 "+Plugin" 恰好给对应基础方法加了同源的 plugin 约束

---

## 五、扰动设计

用于 Base-aug 训练增强 + 所有方法的 robust evaluation：

| 扰动名 | 类型 | 描述 |
|--------|------|------|
| typo | 字符级 | 随机替换/删除/插入字符，rate=0.10 |
| distractor | 语义干扰 | 在问题前插入无关句子（固定候选库） |
| format_rewrite | 格式改写 | 将选项格式从字母标注改为数字/序号等 |

**评测协议**：所有方法在同一套扰动上评测，使用相同随机种子。

---

## 六、Gamma 设计

**Gamma sweep：** q10 / q25 / q50

校准流程：
1. 在 Base-aug checkpoint 上
2. 对扰动验证集（val + typo 扰动）进行推理
3. 计算每个样本的 true-label margin = p(y_true) - max_{k≠y} p(k)
4. 取分位数 q10/q25/q50 作为 gamma

每个 gamma 对应一套独立的 Plugin 训练和评测。

---

## 七、评测指标

必须报告：
1. **Clean Accuracy**
2. **Robust Accuracy**（每种扰动分别报）
3. **Worst-Class Accuracy**
4. **Worst-Class Error**
5. **VWR_gamma**（transition matrix 1-norm 最大列和）
6. **σ_max**（transition matrix spectral norm）
7. **Clean→Robust Drop**（Δ = Clean Acc - Avg Robust Acc）

附加指标：
8. 各类 class-wise accuracy
9. Fragile sample ratio（gate > 0.5 的样本比例）
10. Mean gate value

---

## 八、输出表格

| 表 | 内容 |
|----|------|
| 表 1 | 主结果表：方法 × 扰动 × 全指标 |
| 表 2 | Plugin 增益表：∆robust, ∆worst-class, ∆VWR, ∆σ_max |
| 表 3 | Ablation：Base-aug / +R_spec / +R_stab / +Plugin |
| 图 1 | Gamma sweep 曲线：clean acc / robust acc / worst-class acc / VWR vs. gamma |

---

## 九、超参数默认配置

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| lora_rank | 16 | LoRA 秩 |
| lora_alpha | 32 | LoRA scaling |
| lr | 2e-4 | 学习率 |
| batch_size | 8 | batch size |
| grad_accum | 4 | 梯度累积步数 |
| num_epochs | 5 | 训练轮数 |
| warmup_ratio | 0.1 | warmup 比例 |
| alpha (R_spec) | 0.1 | Plugin R_spec 权重 |
| beta (R_stab) | 0.05 | Plugin R_stab 权重 |
| kappa | 0.5 | gate temperature |
| r3f_lambda | 1.0 | R3F 正则权重 |
| r3f_noise_std | 1e-3 | R3F 嵌入噪声标准差 |
| smart_lambda | 1.0 | SMART 正则权重 |
| smart_epsilon | 1e-3 | SMART 对抗扰动大小 |
| smart_steps | 1 | SMART PGD 步数（效率优先） |
| awp_lambda | 1.0 | AWP 正则权重 |
| awp_adv_lr | 1e-4 | AWP LoRA 权重扰动步长 |
| awp_adv_eps | 1e-4 | AWP LoRA 权重扰动 clip 范围 |

---

## 十、实施优先级

**第一优先级（必须跑通）：**
- Base-clean / Base-aug / Base-aug+Plugin（q25）
- R3F / R3F+Plugin
- SMART / SMART+Plugin
- Gamma sweep（q10/q25/q50）对 Plugin
- 主结果表

**第二优先级（尽量完成）：**
- AWP / AWP+Plugin
- Ablation（R_spec only, R_stab only）

**Seeds：** 42, 123（代码结构支持扩展到 3 个）

---

## 十一、目录结构

```
/LOCAL2/zhuoyun/PAC_robust/v2/
├── config.py                  # 全局配置
├── data.py                    # 数据加载（ARC）
├── perturb.py                 # 文本扰动
├── models.py                  # 模型加载（LoRA + 4-bit）
├── plugin.py                  # Plugin 核心（margin, gate, T, R_spec, R_stab）
├── trainers.py                # 所有 Trainer（Base/R3F/SMART/AWP + PluginWrapper）
├── eval.py                    # 评测模块
├── gamma_calibrate.py         # Gamma 校准
├── scripts/
│   ├── run_experiment.py      # 单次实验入口（一个 method × seed）
│   ├── run_all.py             # 编排所有实验
│   ├── gamma_sweep.py         # Gamma sweep
│   └── aggregate.py           # 汇总结果，生成表格和曲线数据
└── results/                   # CSV / JSON / plots 输出目录
```
