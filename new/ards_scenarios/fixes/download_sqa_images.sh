#!/bin/bash
# download_sqa_images.sh
# 下载 ScienceQA 测试集图像
# 来源: HuggingFace datasets derek-thomas/ScienceQA

set -e
SQA_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa"
echo "=== 下载 ScienceQA 测试集图像 ==="
echo "目标路径: $SQA_DIR"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ards

export HF_HOME=/LOCAL2/zhuoyun/hf_cache
export TRANSFORMERS_CACHE=/LOCAL2/zhuoyun/hf_cache

# 方法1: 用 huggingface_hub 下载 (推荐)
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os, shutil

sqa_dir = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa"
os.makedirs(sqa_dir, exist_ok=True)

print("Downloading ScienceQA dataset from HuggingFace...")
local_path = snapshot_download(
    repo_id="derek-thomas/ScienceQA",
    repo_type="dataset",
    local_dir="/LOCAL2/zhuoyun/hf_cache/datasets/ScienceQA",
    ignore_patterns=["*.parquet", "*.arrow"],
)
print(f"Downloaded to: {local_path}")

# 检查目录结构
import os
for root, dirs, files in os.walk(local_path):
    depth = root.replace(local_path, '').count(os.sep)
    if depth < 3:
        print('  ' * depth + os.path.basename(root) + '/')
    if depth == 0:
        for f in files[:5]:
            print('  ' + f)
PYEOF

echo "=== 完成 ==="
