#!/bin/bash

# ==================== VGGT KITTI Scale Head Training - Quick Start ====================
# 
# 使用方法：
#   bash quickstart_training.sh
#   或
#   bash quickstart_training.sh --epochs 50 --batch_size 16
#

set -e

echo "=================================="
echo "VGGT KITTI Scale Head Training"
echo "=================================="
echo ""

# 检查Python环境
echo "[Step 1] 检查Python环境..."
python --version
echo "✓ Python环境正常"
echo ""

# 检查PyTorch
echo "[Step 2] 检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')" 2>/dev/null || { echo "✗ PyTorch不可用"; exit 1; }
echo "✓ PyTorch环境正常"
echo ""

# 检查配置文件
echo "[Step 3] 检查配置文件..."
if [ ! -f "configs/train_scale_head_kitti.yaml" ]; then
    echo "✗ 配置文件不存在: configs/train_scale_head_kitti.yaml"
    exit 1
fi
echo "✓ 配置文件存在"
echo ""

# 检查数据集
echo "[Step 4] 检查KITTI数据集..."
DATA_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/train_scale_head_kitti.yaml'))['data']['data_dir'])" 2>/dev/null)
if [ ! -d "$DATA_DIR" ]; then
    echo "✗ KITTI数据集目录不存在: $DATA_DIR"
    echo "  请编辑 configs/train_scale_head_kitti.yaml 并设置正确的data_dir"
    exit 1
fi
echo "✓ KITTI数据集合法: $DATA_DIR"
echo ""

# 检查预训练模型
echo "[Step 5] 检查预训练模型..."
CKPT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('configs/train_scale_head_kitti.yaml'))['model']['ckpt_path'])" 2>/dev/null)
if [ ! -f "$CKPT_PATH" ]; then
    echo "✗ 预训练模型不存在: $CKPT_PATH"
    echo "  请编辑 configs/train_scale_head_kitti.yaml 并设置正确的ckpt_path"
    exit 1
fi
echo "✓ 预训练模型存在: $CKPT_PATH"
echo ""

# 建立输出目录
echo "[Step 6] 建立输出目录..."
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/train_scale_head_kitti.yaml'))['training']['output_dir'])" 2>/dev/null)
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"
echo ""

# 显示训练配置
echo "[Step 7] 显示训练配置..."
python << 'EOF'
import yaml

with open('configs/train_scale_head_kitti.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("数据配置:")
print(f"  数据集: {config['data']['data_dir']}")
print(f"  训练样本数: {config['data']['train_num']}")
print(f"  验证样本数: {config['data']['val_num']}")
print(f"  Batch大小: {config['data']['batch_size']}")
print("")
print("模型配置:")
print(f"  Image Size: {config['model']['img_size']}")
print(f"  Embed Dim: {config['model']['embed_dim']}")
print(f"  预训练模型: {config['model']['ckpt_path']}")
print("")
print("训练配置:")
print(f"  总Epoch数: {config['training']['epochs']}")
print(f"  学习率: {config['training']['lr']}")
print(f"  Weight Decay: {config['training']['weight_decay']}")
print(f"  早停耐心值: {config['training']['early_stopping_patience']}")
print("")
EOF

echo "✓ 配置显示完成"
echo ""

# 开始训练
echo "=================================="
echo "开始训练..."
echo "=================================="
echo ""

# 解析命令行参数
ARGS=""
for arg in "$@"; do
    ARGS="$ARGS --$arg"
done

# 如果有参数则传递给训练脚本
if [ -z "$ARGS" ]; then
    python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml
else
    echo "使用自定义参数: $ARGS"
    python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml $ARGS
fi

echo ""
echo "=================================="
echo "✓ 训练完成！"
echo "=================================="
echo ""
echo "输出位置: $OUTPUT_DIR"
echo "最优模型: $OUTPUT_DIR/checkpoints/scale_head_best.pt"
echo "训练日志: $OUTPUT_DIR/training.log"
echo ""
echo "查看日志："
echo "  cat $OUTPUT_DIR/training.log"
echo ""
