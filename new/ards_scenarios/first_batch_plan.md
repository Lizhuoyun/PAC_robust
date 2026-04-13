# 第一批实验计划

**创建时间**: 2026-04-10  
**原则**: 最少工作量、最高复用价值、先跑通 eval 闭环再扩展训练

---

## 一、选定的第一批任务

### 第一优先级: ScienceQA（eval + 攻击评测）

**理由**:
1. 注释文件已存在于仓库（`llava_test_CQM-A.json`, `pid_splits.json`, `problems.json`）
2. 脚本完整：clean eval + PA + SA 三种评测，是本项目最完整的任务
3. 选择题格式 → 天然离散输出，与后续 Plugin-LoRA 最契合
4. 数据量适中：test ~4.2k 样本
5. Full-data baseline 直接用 `liuhaotian/llava-v1.5-7b`（无需训练）

**需要准备的东西**:
- [ ] 下载 `liuhaotian/llava-v1.5-7b` (~14GB)
- [ ] 下载 ScienceQA test 图像 (~2.8GB from GitHub/HF)
- [ ] 搭建 `ards` conda 环境

**期望产出**:
- clean accuracy on ScienceQA test
- PA accuracy（选项字母扰动 ABCDE→随机）
- SA accuracy（符号攻击 ABCDE→QWERT）
- 这三个数字作为 full-data baseline（与论文 Table 1 对比）

---

### 第二优先级: GQA（eval）

**理由**:
1. 评测脚本完整（multi-GPU），支持 PA 攻击
2. 开放答案补充选择题之外的维度
3. GQA-OOD 与 GQA 共享图像，两个同时下载效率高
4. 图像大但只需 testdev_balanced 子集

**需要准备的东西**:
- [ ] 下载 GQA testdev_balanced 问题 JSON
- [ ] 下载 GQA images (~20GB，与 GQA-OOD 共享)
- [ ] 运行 `create_llava_file.py` 生成 llava 格式

---

### 第三优先级: TextVQA（eval）

**理由**:
1. 评测脚本完整
2. OCR 相关，覆盖 GQA/ScienceQA 之外的鲁棒性维度
3. 数据较小（~8GB）

---

### 暂时搁置

- **SEED-Bench**: 数据太重（含视频），等有余力再下载
- **MMBench**: 需先修复缺失的 `model_vqa_mmbench.py`
- **GQA-OOD**: 跟 GQA 一起做，共用图像

---

## 二、执行顺序

```
Phase 1: 环境搭建（并行进行）
├── conda create -n ards python=3.10
├── pip install torch==2.1.2 torchvision==0.16.2
├── cd /LOCAL2/zhuoyun/PAC_robust/ARDS && pip install -e ".[train]"
└── pip install flash-attn --no-build-isolation

Phase 2: 数据与模型下载（并行进行）
├── HF 下载: liuhaotian/llava-v1.5-7b
├── HF/GitHub 下载: ScienceQA test images
└── 下载: GQA testdev_balanced + images（可后台进行）

Phase 3: ScienceQA eval 闭环（串行）
├── Step 1: clean eval
│   └── python -m llava.eval.model_vqa_science ...
├── Step 2: eval 指标
│   └── python llava/eval/eval_science_qa.py ...
├── Step 3: PA eval
│   └── python -m llava.eval.model_vqa_science_option_attack ... --options A B C D E
└── Step 4: SA eval
    └── python -m llava.eval.model_vqa_science_option_attack ... --options Q W E R T

Phase 4: GQA + TextVQA eval（并行多卡）
├── GQA: CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/v1_5/eval/gqa.sh
└── TextVQA: CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
```

---

## 三、目录结构规划

```
/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/
├── ARDS_reuse_audit.md           # 仓库审计（本文件同级）
├── first_batch_plan.md           # 本文件
├── reusable_baselines_summary.md # 实验结果汇总（运行后填充）
├── next_step_for_plugin.md       # Plugin 接入说明
├── checkpoints/                  # baseline checkpoint 路径（链接或记录）
├── data/                         # 数据下载目录（或软链接）
│   ├── scienceqa/
│   ├── gqa/
│   ├── gqa_ood/
│   └── textvqa/
├── eval_outputs/                 # 各任务评测输出
│   ├── scienceqa/
│   │   ├── clean/
│   │   ├── pa_attack/
│   │   └── sa_attack/
│   ├── gqa/
│   └── textvqa/
├── logs/                         # 训练/评测日志
├── selected_data/                # ARDS 选择结果（运行后）
└── fixes/                        # 最小修复脚本
    ├── env_setup.sh
    ├── download_sqa_images.sh
    ├── run_sqa_eval.sh
    └── run_gqa_eval.sh
```

---

## 四、为何不先做全量训练 baseline

1. `llava-v1.5-7b` **本身就是** ARDS 论文中的 full-data baseline（mix665k 全量训练）
2. 重新训练需要：
   - mix665k 注释文件 + COCO/GQA/OCR-VQA/TextVQA/VG 图像（数百 GB）
   - 预训练 connector `llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5`
   - DeepSpeed + 多卡，训练时间约 2-3 天
3. **直接用 HF 上的 7b 模型作为 eval baseline 更高效**
4. 后续 LoRA / Plugin-LoRA 实验可以基于这个 7b 模型做 **任务特定微调**

---

## 五、关于 ARDS LoRA baseline

ARDS 提供了 Google Drive 链接下载 ARDS-selected LoRA checkpoint，这是与 full-data baseline 的核心对比点。  
建议在 ScienceQA baseline eval 跑通后，再下载 ARDS LoRA checkpoint 做对比。

下载链接（来自 README）:
- Full-data LoRA: https://drive.google.com/drive/folders/1KBgiB4AcvIgUkfTXs_wiZSWXzMaIxRVN
- ARDS LoRA: https://drive.google.com/drive/folders/1VvRi-x61GJ0UXmXUYjsiyIYvD744qaZ2
- ARDS 选择结果: https://drive.google.com/file/d/1rgzC3-aO-AgX08452HrlyxHWnldrjm4o
