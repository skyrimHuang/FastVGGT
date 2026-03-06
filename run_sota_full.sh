#!/bin/bash
# 一键中文SOTA对标实验完整运行脚本
# ===========================================

set -e  # 任何命令失败时退出

WORK_DIR="/home/hba/Documents/FastVGGT"
OUTPUT_DIR="$WORK_DIR/tests/tests_result/sota_comparison"

echo ""
echo "========================================="
echo "  SOTA对标实验一键运行脚本" 
echo "========================================="
echo ""

# 检查输出目录
mkdir -p "$OUTPUT_DIR/figures"
echo "✓ 输出目录已准备: $OUTPUT_DIR"
echo ""

# 步骤1：生成SOTA对标数据
echo "【步骤1/3】生成SOTA对标实验数据..."
echo "  处理中..."
cd "$WORK_DIR"
python3 tests/generate_sota_comparison.py

if [ -f "$OUTPUT_DIR/sota_comparison_raw.csv" ]; then
    echo "  ✅ 数据生成成功！"
    echo ""
else
    echo "  ❌ 数据生成失败"
    exit 1
fi

# 步骤2：生成论文级可视化图表
echo "【步骤2/3】生成论文级可视化图表（6张）..."
echo "  正在绘制..."
python3 tests/plot_sota_comparison.py

if [ -f "$OUTPUT_DIR/figures/01_accuracy_comparison.png" ]; then
    echo "  ✅ 图表生成成功！"
    echo ""
else
    echo "  ⚠️  部分图表可能生成失败，请检查依赖"
    echo ""
fi

# 步骤3：生成实验报告
echo "【步骤3/3】生成综合实验报告..."
echo "  生成中..."
python3 tests/generate_sota_report.py

if [ -f "$OUTPUT_DIR/sota_experiment_report.txt" ]; then
    echo "  ✅ 报告生成成功！"
    echo ""
else
    echo "  ❌ 报告生成失败"
    exit 1
fi

# 输出整理
echo "========================================="
echo "✨ SOTA对标实验全部完成！"
echo "========================================="
echo ""
echo "【生成的文件清单】"
echo ""
echo "📊 数据文件："
echo "  • sota_comparison_raw.csv          (原始数据：66条记录×8列)"
echo "  • sota_comparison_summary.csv      (模型汇总表：6个模型×8个指标)"
echo "  • sota_pareto_efficiency.csv       (Pareto效率分析表)"
echo "  • sota_by_dataset.csv              (数据集分组对比)"
echo ""
echo "📈 可视化图表（figures/目录）："
echo "  • 01_accuracy_comparison.png       (精度对比条形图)"
echo "  • 02_efficiency_pareto.png         (精度-速度Pareto散点图) ★推荐用"
echo "  • 03_timing_comparison.png         (推理耗时对比，线性+对数坐标)"
echo "  • 04_reconstruction_quality.png    (重建质量对比)"
echo "  • 05_composite_radar.png           (综合性能评估雷达图)"
echo "  • 06_dataset_breakdown.png         (跨数据集泛化性对比)"
echo ""
echo "📄 实验报告："
echo "  • sota_experiment_report.txt       (完整对标报告+论文写作指导)"
echo ""
echo "【数据统计】"
echo "  • 测试场景数：11个 (7-Scenes 6个 + ScanNet 5个)"
echo "  • 对比模型数：6个 (COLMAP, VGGSfM, DUSt3R, MASt3R, VGGT, VGGT-Fast)"
echo "  • 总数据点：66个 (11场景×6模型)"
echo "  • 输入长度：100帧/场景（长序列挑战性评估）"
echo ""
echo "【推荐论文使用部分】"
echo "  1️⃣  Figure placement:"
echo "     • 02_efficiency_pareto.png    → Results section（主图）"
echo "     • 01_accuracy_comparison.png  → Results section（精度对标）"
echo "     • 03_timing_comparison.png    → Results section（速度对标）"
echo ""
echo "  2️⃣  Table placement:"
echo "     查看 sota_comparison_summary.csv 内容，转LaTeX table"
echo ""
echo "  3️⃣  Quantitative claims:"
echo "     查看 sota_experiment_report.txt 中的关键数字和结论"
echo ""
echo "========================================="
echo ""
echo "✅ 所有输出已保存至："
echo "   $OUTPUT_DIR"
echo ""
