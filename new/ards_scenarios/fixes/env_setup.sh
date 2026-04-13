#!/bin/bash
# env_setup.sh
# 为 ARDS 创建专用 conda 环境
# 需要先激活 conda base: source /opt/anaconda3/etc/profile.d/conda.sh

set -e
echo "=== ARDS 环境安装脚本 ==="

# 激活 conda
source /opt/anaconda3/etc/profile.d/conda.sh

# 创建/激活 ards env（若已存在跳过创建）
conda activate ards 2>/dev/null || conda create -n ards python=3.10 -y && conda activate ards

echo "Python: $(python --version)"

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装 ARDS (LLaVA 核心 + 训练依赖)
cd /LOCAL2/zhuoyun/PAC_robust/ARDS
pip install -e .
pip install -e ".[train]"

# 安装 flash-attn（必须在 torch 安装后）
pip install flash-attn --no-build-isolation

# 安装 faiss-gpu（数据选择时需要）
pip install faiss-gpu==1.8.0

# 其他常用工具
pip install wandb matplotlib seaborn

echo "=== 安装完成 ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import peft; print('peft:', peft.__version__)"
