#!/bin/bash
# download_sqa_images_github.sh
# 从 ScienceQA GitHub 下载测试集图像（另一种方式）
# Requires: git lfs installed

SQA_DIR="/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa"

echo "Downloading ScienceQA images from GitHub..."
# The ScienceQA images are available via HuggingFace:
# https://huggingface.co/datasets/derek-thomas/ScienceQA

# Alternative: download directly from the official dataset
# The test images are ~2.8GB

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ards

python3 << 'PYEOF'
import os
from datasets import load_dataset

print("Loading ScienceQA dataset (test split)...")
dataset = load_dataset("derek-thomas/ScienceQA", split="test",
                       cache_dir="/LOCAL2/zhuoyun/hf_cache")
print(f"Test split: {len(dataset)} samples")
print("Columns:", dataset.column_names)

# Save images to the expected directory
test_img_dir = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/scienceqa/test"
os.makedirs(test_img_dir, exist_ok=True)

saved = 0
for i, sample in enumerate(dataset):
    if sample.get('image') is not None:
        img_path = os.path.join(test_img_dir, f"{sample['id']}")
        os.makedirs(img_path, exist_ok=True)
        sample['image'].save(os.path.join(img_path, "image.png"))
        saved += 1
    if (i+1) % 100 == 0:
        print(f"  Processed {i+1}/{len(dataset)}, saved {saved} images")

print(f"Done! Saved {saved} images to {test_img_dir}")
PYEOF
