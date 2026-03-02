#!/bin/bash
# Token分割方法对比 - 完整执行pipeline
# ===========================================================
# 步骤：
# 1. 运行快速测试生成数据
# 2. 生成中文本地化的可视化
# 3. 对比点云重建结果
# 4. 输出实验结果

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Token分割方法对比 - 完整分析Pipeline                          ║"
echo "║  包括：中文可视化 + COLMAP点云对比                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检查环境
conda_env="fastvggt"

echo "🔍 检查环境..."
if ! conda info --envs | grep -q "^$conda_env"; then
    echo "❌ conda环境 '$conda_env' 不存在"
    echo "请先创建环境: conda create -n fastvggt python=3.10 pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
    exit 1
fi

echo "✓ 环境检查完成"
echo ""

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $conda_env

cd /home/hba/Documents/FastVGGT

# Step 1: 快速测试（生成示例数据）
echo "════════════════════════════════════════════════════════════════"
echo "Step 1: 运行快速测试生成数据"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  预计时间: 30-60分钟（2个场景）"
echo ""

TEST_FRAMES=${1:-50}
TEST_SCENES=${2:-2}
DATA_ROOT=${3:-/data/ScanNet_test/scans}

echo "参数:"
echo "  • 帧数: $TEST_FRAMES"
echo "  • 场景数: $TEST_SCENES"
echo "  • 数据路径: $DATA_ROOT"
echo ""

# 检查数据路径
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  数据路径不存在: $DATA_ROOT"
    echo "请将ScanNet数据放在该目录，或通过第三个参数指定"
    echo "跳过实验运行，仅使用示例数据..."
fi

# 运行实验
echo "🚀 启动实验..."
python tests/run_token_split_ablation.py \
    --frames $TEST_FRAMES \
    --num_scenes $TEST_SCENES \
    --data_root "$DATA_ROOT" \
    --model_path ckpt/model_tracker_fixed_e20.pt \
    --output_dir tests/tests_result/token_split_ablation

echo "✅ 实验完成！"
echo ""

# Step 2: 生成中文本地化可视化
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: 生成中文本地化可视化"
echo "════════════════════════════════════════════════════════════════"
echo ""

RESULTS_JSON="tests/tests_result/token_split_ablation/token_split_ablation_summary.json"

if [ ! -f "$RESULTS_JSON" ]; then
    echo "❌ 结果文件不存在: $RESULTS_JSON"
    exit 1
fi

echo "📊 生成图表..."
python tests/analyze/plot_token_split_cn.py \
    --input_json "$RESULTS_JSON" \
    --output_dir tests/tests_result/token_split_ablation/figures_cn

echo "✅ 中文可视化完成！"
echo ""

# Step 3: COLMAP点云对比（可选）
echo "════════════════════════════════════════════════════════════════"
echo "Step 3: COLMAP点云重建对比（可选）"
echo "════════════════════════════════════════════════════════════════"
echo ""

BASELINE_COLMAP=${4:-}
PROPOSED_COLMAP=${5:-}

if [ -n "$BASELINE_COLMAP" ] && [ -n "$PROPOSED_COLMAP" ]; then
    if [ -d "$BASELINE_COLMAP" ] && [ -d "$PROPOSED_COLMAP" ]; then
        echo "🔄 对比COLMAP重建..."
        python tests/compare_reconstruction_colmap.py \
            --baseline_colmap "$BASELINE_COLMAP" \
            --proposed_colmap "$PROPOSED_COLMAP" \
            --output_dir tests/tests_result/reconstruction_comparison
        echo "✅ 点云对比完成！"
    else
        echo "⚠️  COLMAP目录不存在，跳过点云对比"
    fi
else
    echo "ℹ️  未提供COLMAP路径，跳过点云对比"
    echo ""
    echo "如果需要点云对比，请使用以下参数运行此脚本："
    echo "  $0 <frames> <scenes> <data_root> <baseline_colmap> <proposed_colmap>"
    echo ""
    echo "示例:"
    echo "  $0 50 2 /data/ScanNet ./colmap/baseline/sparse ./colmap/proposed/sparse"
fi

echo ""

# 总结
echo "════════════════════════════════════════════════════════════════"
echo "✨ Pipeline执行完成！"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "生成的文件："
echo ""
echo "📊 中文本地化可视化:"
echo "  • figure_3_12_cn_token_split_comparison.png  - 主对比图（三种方法）"
echo "  • figure_supp_cn_inference_time.png          - 推理时间图"
echo "  • figure_metrics_heatmap_cn.png              - 全指标热力图"
echo ""
echo "📁 文件位置:"
echo "  → tests/tests_result/token_split_ablation/figures_cn/"
echo ""

if [ -n "$BASELINE_COLMAP" ] && [ -n "$PROPOSED_COLMAP" ]; then
    echo "☁️  点云对比文件:"
    echo "  • baseline_grid_based.ply                  - 基线点云"
    echo "  • proposed_norm_guided.ply                 - 本文方法点云"
    echo "  • reconstruction_comparison.ply            - 双色对比点云"
    echo "  • trajectory_baseline.ply                  - 基线摄像机轨迹"
    echo "  • trajectory_proposed.ply                  - 本文摄像机轨迹"
    echo ""
    echo "📁 文件位置:"
    echo "  → tests/tests_result/reconstruction_comparison/"
    echo ""
    echo "📍 查看提示:"
    echo "  • CloudCompare: 查看和编辑PLY点云"
    echo "  • Meshlab: 点云处理和分析"
    echo "  • COLMAP GUI: 查看完整的SfM重建"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
