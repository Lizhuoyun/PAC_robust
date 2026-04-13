# ARDS Selection Pipeline Notes

## Pipeline Overview

ARDS selection 是一个 4 步流程，最终从大训练集中挑出对 worst-case robustness 最有用的子集。

```
Step A: get_train_repr.py       →  提取每个训练样本的 representation (3 GPU 并行)
Step A2: merge_repr_grad_files.py → 合并分片 repr 为 all_orig.pt
Step B: cluster_subgroup.py     →  K-means 聚类，按 format 分 subgroup
Step C: build_worst_subgroup.py →  在每个 subgroup 内找 worst-case 样本
Step D: matching_worst_subgroup.py → 用 influence function 对训练集排序，选出 top-k
```

## ScienceQA 实际执行记录

### Step A: Representation 提取

- 脚本: `llava/data/get_train_repr.py`
- 修复: 删除 `load_pretrained_model_deepspeed` 导入，删除 `merge_lora=False` 参数，关闭 flash_attention_2
- 模型: pre-trained llava-v1.5-7b（直接使用，无需 warmup）
- batch_size: 2（output_attentions=True 导致显存大量占用，16 会 OOM）
- 3 GPU 并行，每张 GPU ~4 分钟，共 12726 条数据
- 输出: 639 个 .pt 分片文件

### Step A2: Merge

- 输出: `sqa_selection/reps/weighted_attn/all_orig.pt` (12726 条，dim=8192)

### Step B: K-means 聚类

- 使用 CPU faiss（GPU 版本在高维度上会卡死）
- K=10 clusters，收敛于 30 次迭代
- Cluster 大小: [2049, 1989, 1805, 1430, 1390, 1274, 979, 646, 594, 570]
- 输出: `kmeans.json`

### Step C: Worst Subgroup

- ScienceQA 全部是 fmt_choice 格式
- 每个 cluster 采样 50 个代表样本
- 共 500 个 worst subgroup 样本
- 输出: `worst_group_samples.json`

### Step D: Influence Score + Selection

- cosine similarity 计算 + 按 subgroup 加权
- Score 范围: [0.494, 0.784], mean=0.717
- 输出:
  - `scienceqa_selected_scores.json` (12226 条 scores)
  - `scienceqa_selected_subset.json` (3817 条, top 30%, 默认)
  - `scienceqa_selected_subset_top50.json` (6363 条)
  - `scienceqa_selected_subset_top70.json` (8908 条)
  - `scienceqa_selected_ids_top30.json` / `top50` / `top70`

## 代码修改记录

1. `llava/data/get_train_repr.py` line 31: 删除 `load_pretrained_model_deepspeed` 导入
2. `llava/data/get_train_repr.py` line 830: 删除 `merge_lora=False` 参数
3. `llava/data/get_train_repr.py` line 900: `flash_attention_2` → `None`
4. `llava/data/get_train_loss.py` line 31: 同 #1
5. `llava/data/get_train_loss.py` line 810: 同 #2
6. 新增 `playground/data/eval/scienceqa/llava_train_QCM-A_globalid.json` (训练数据 + global_id)

## 输出文件清单

```
sqa_selection/
├── reps/weighted_attn/
│   ├── 3_0/ 3_1/ 3_2/     (分片 repr)
│   ├── all_orig.pt          (normalized, 12726 x 8192)
│   ├── all_unormalized.pt   (unnormalized)
│   └── kmeans.json          (cluster assignments)
├── worst_group_samples.json (500 worst samples)
├── scienceqa_selected_scores.json (12226 influence scores)
├── scienceqa_selected_subset.json (3817, default top30%)
├── scienceqa_selected_subset_top30.json
├── scienceqa_selected_subset_top50.json
├── scienceqa_selected_subset_top70.json
├── scienceqa_selected_ids_top30.json
├── scienceqa_selected_ids_top50.json
├── scienceqa_selected_ids_top70.json
├── repr_gpu0.log / gpu1 / gpu2
└── run_sqa_cluster_and_select.py
```
