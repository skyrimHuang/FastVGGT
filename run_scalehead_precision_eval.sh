#!/bin/bash
# ScaleHead Precision Evaluation Run Script

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ScaleHead 双目绝对尺度恢复精度验证（3.3.2节）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 配置
DATA_DIR="${1:-/home/hba/Documents/Dataset/KITTI}"
CKPT_BACKBONE="${2:-ckpt/model_tracker_fixed_e20.pt}"
CKPT_SCALEHEAD="${3:-outputs/kitti_scale_head/checkpoints/scale_head_best.pt}"
OUTPUT_DIR="${4:-outputs/scalehead_precision_eval}"

echo "📁 Configuration:"
echo "  Data Dir:         $DATA_DIR"
echo "  Backbone Ckpt:    $CKPT_BACKBONE"
echo "  ScaleHead Ckpt:   $CKPT_SCALEHEAD"
echo "  Output Dir:       $OUTPUT_DIR"
echo ""

# 验证文件存在
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$CKPT_BACKBONE" ]; then
    echo "⚠ Warning: Backbone checkpoint not found: $CKPT_BACKBONE"
fi

if [ ! -f "$CKPT_SCALEHEAD" ]; then
    echo "⚠ Warning: ScaleHead checkpoint not found: $CKPT_SCALEHEAD"
fi

echo "🚀 Starting experiment..."
echo ""

cd /home/hba/Documents/FastVGGT

python eval_scalehead_precision_comparison.py \
  --data_dir "$DATA_DIR" \
  --ckpt_path "$CKPT_BACKBONE" \
  --scale_head_ckpt "$CKPT_SCALEHEAD" \
  --output_dir "$OUTPUT_DIR" \
  --train_num 120 \
  --val_num 50 \
  --merging 0 \
  --merge_ratio 0.9 \
  --device cuda

echo ""
echo "✅ Experiment completed!"
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
