#!/usr/bin/env python3
"""
对比 Baseline (Grid-based) vs Improved (Norm-Guided) 的评估结果

Usage:
    python tests/compare_norm_guided.py
"""

import json
from pathlib import Path
import sys

def load_metrics(result_dir, input_frame=100):
    """加载平均指标"""
    metrics_file = Path(result_dir) / f"input_frame_{input_frame}" / "average_metrics.json"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def compute_improvement(baseline_val, improved_val, metric_name):
    """计算改进百分比（正值=改进，负值=退化）"""
    if baseline_val == 0:
        return 0.0
    
    # 对于误差类指标（CD, ATE, ARE等），越小越好
    if metric_name in ['chamfer_distance', 'ate', 'are', 'rpe_rot', 'rpe_trans']:
        improvement = (baseline_val - improved_val) / baseline_val * 100
    else:
        # 对于速度等，值越大越好（但这里inference_time越小越好）
        if metric_name == 'inference_time_ms':
            improvement = (baseline_val - improved_val) / baseline_val * 100
        else:
            improvement = (improved_val - baseline_val) / baseline_val * 100
    
    return improvement

def print_comparison(baseline_metrics, improved_metrics):
    """打印对比结果表格"""
    print("\n" + "="*80)
    print("NORM-GUIDED ANCHORING + THRESHOLD-GATED ANTI-COLLAPSE COMPARISON")
    print("="*80)
    
    # 定义指标顺序和格式
    metrics_info = [
        ('chamfer_distance', 'Chamfer Distance', '.6f'),
        ('ate', 'Absolute Trajectory Error', '.6f'),
        ('are', 'Absolute Rotation Error', '.6f'),
        ('rpe_rot', 'Relative Pose Error (Rot)', '.6f'),
        ('rpe_trans', 'Relative Pose Error (Trans)', '.6f'),
        ('inference_time_ms', 'Inference Time (ms)', '.2f'),
    ]
    
    print(f"\n{'Metric':<35} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
    print("-"*80)
    
    for metric_key, metric_name, fmt_str in metrics_info:
        if metric_key in baseline_metrics and metric_key in improved_metrics:
            baseline_val = baseline_metrics[metric_key]
            improved_val = improved_metrics[metric_key]
            improvement = compute_improvement(baseline_val, improved_val, metric_key)
            
            # 格式化输出
            baseline_str = f"{baseline_val:{fmt_str}}"
            improved_str = f"{improved_val:{fmt_str}}"
            
            # 改进百分比符号
            if improvement > 0:
                change_str = f"↓ {improvement:.2f}% ✓"  # 改进（误差减少）
            elif improvement < 0:
                change_str = f"↑ {abs(improvement):.2f}% ✗"  # 退化（误差增加）
            else:
                change_str = "= 0.00%"
            
            print(f"{metric_name:<35} {baseline_str:<15} {improved_str:<15} {change_str:<15}")
    
    print("="*80)
    
    # 输出关键发现
    print("\nKEY FINDINGS:")
    cd_improvement = compute_improvement(
        baseline_metrics['chamfer_distance'], 
        improved_metrics['chamfer_distance'],
        'chamfer_distance'
    )
    
    if cd_improvement > 10:
        print(f"✓ Chamfer Distance improved by {cd_improvement:.2f}% - SIGNIFICANT IMPROVEMENT")
    elif cd_improvement > 0:
        print(f"✓ Chamfer Distance improved by {cd_improvement:.2f}% - Modest improvement")
    else:
        print(f"✗ Chamfer Distance worsened by {abs(cd_improvement):.2f}% - Needs investigation")
    
    time_improvement = compute_improvement(
        baseline_metrics['inference_time_ms'],
        improved_metrics['inference_time_ms'],
        'inference_time_ms'
    )
    
    if time_improvement > 0:
        print(f"✓ Inference time reduced by {time_improvement:.2f}%")
    elif abs(time_improvement) < 5:
        print(f"≈ Inference time roughly the same ({time_improvement:+.2f}%)")
    else:
        print(f"⚠ Inference time increased by {abs(time_improvement):.2f}%")

def main():
    # 定义结果目录
    baseline_dir = "tests/tests_result/baseline_grid_5scenes"
    improved_dir = "tests/tests_result/improved_norm_guided_5scenes"
    input_frame = 100
    
    print("Loading metrics...")
    baseline_metrics = load_metrics(baseline_dir, input_frame)
    improved_metrics = load_metrics(improved_dir, input_frame)
    
    if baseline_metrics is None:
        print(f"ERROR: Baseline metrics not found at {baseline_dir}")
        print("Please run baseline test first:")
        print("  python eval/eval_scannet.py --merging 0 --merge_ratio 0.9 \\")
        print("    --input_frame 100 --num_scenes 5 \\")
        print("    --output_path ./tests/tests_result/baseline_grid_5scenes")
        return 1
    
    if improved_metrics is None:
        print(f"ERROR: Improved metrics not found at {improved_dir}")
        print("Please run improved test first:")
        print("  python eval/eval_scannet.py --merging 0 --merge_ratio 0.9 \\")
        print("    --use_norm_guided --merge_threshold 0.85 \\")
        print("    --input_frame 100 --num_scenes 5 \\")
        print("    --output_path ./tests/tests_result/improved_norm_guided_5scenes")
        return 1
    
    # 打印对比结果
    print_comparison(baseline_metrics, improved_metrics)
    
    # 保存对比结果到JSON
    comparison_result = {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "improvements": {
            key: compute_improvement(baseline_metrics[key], improved_metrics[key], key)
            for key in baseline_metrics.keys()
            if key in improved_metrics
        }
    }
    
    output_file = Path("tests/tests_result/comparison_norm_guided.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison_result, f, indent=4)
    print(f"\n✓ Comparison results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
