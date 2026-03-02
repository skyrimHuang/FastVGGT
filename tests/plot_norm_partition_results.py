"""
Norm分区比率优化结果可视化
==========================

读取分区比率优化测试的CSV结果文件，生成中文标注的分析图表。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
rcParams['axes.unicode_minus'] = False


def load_results(csv_path: str) -> pd.DataFrame:
    """加载CSV结果文件。"""
    df = pd.read_csv(csv_path)
    return df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """创建多个分析图表。"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 精度vs压缩率 (Pareto前沿)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df['compression'], df['cd'], 
                        s=200, alpha=0.7, c=range(len(df)), cmap='viridis', 
                        edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for idx, row in df.iterrows():
        ax.annotate(f"P:{row['protected']:.0%}\nD:{row['dst']:.0%}\nS:{row['src']:.0%}",
                   xy=(row['compression'], row['cd']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('压缩率 (倍)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CD - Chamfer距离 (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Norm分区比率优化: 精度vs压缩率', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 压缩率越低越好，放在右边
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_compression.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: accuracy_vs_compression.png")
    plt.close()
    
    # 2. 所有指标的权衡分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Norm分区比率优化: 各指标对比', fontsize=14, fontweight='bold')
    
    # CD (Chamfer距离)
    ax = axes[0, 0]
    bars = ax.bar(range(len(df)), df['cd'], color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax.set_ylabel('CD (cm)', fontsize=11, fontweight='bold')
    ax.set_title('几何精度 (CD越低越好)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{row['dst']:.0%}" for _, row in df.iterrows()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ATE (位置误差)
    ax = axes[0, 1]
    bars = ax.bar(range(len(df)), df['ate'], color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax.set_ylabel('ATE (米)', fontsize=11, fontweight='bold')
    ax.set_title('轨迹累积误差 (ATE越低越好)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{row['dst']:.0%}" for _, row in df.iterrows()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ARE (旋转误差)
    ax = axes[1, 0]
    bars = ax.bar(range(len(df)), df['are'], color='#95E1D3', alpha=0.8, edgecolor='black')
    ax.set_ylabel('ARE (度)', fontsize=11, fontweight='bold')
    ax.set_title('旋转误差 (ARE越低越好)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{row['dst']:.0%}" for _, row in df.iterrows()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 推理时间
    ax = axes[1, 1]
    bars = ax.bar(range(len(df)), df['time_ms'], color='#FFD93D', alpha=0.8, edgecolor='black')
    ax.set_ylabel('时间 (毫秒)', fontsize=11, fontweight='bold')
    ax.set_title('推理时间 (时间越短越好)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{row['dst']:.0%}" for _, row in df.iterrows()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # X轴标签: Dst分区比率
    fig.text(0.5, 0.02, 'Dst分区比率 (%)', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: metrics_comparison.png")
    plt.close()
    
    # 3. 相对改进百分比 (相对于基线)
    baseline_cd = df.iloc[0]['cd']
    improvements = ((baseline_cd - df['cd']) / baseline_cd * 100).tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B' if x < 0 else '#51CF66' for x in improvements]
    bars = ax.bar(range(len(df)), improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加百分比标签
    for idx, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('CD改进百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'相对基线(0.10/0.10/0.80)的几何精度改进', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"P:{row['protected']:.0%}\nD:{row['dst']:.0%}\nS:{row['src']:.0%}" 
                         for _, row in df.iterrows()], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_vs_baseline.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: improvement_vs_baseline.png")
    plt.close()
    
    # 4. 多目标优化权衡分析 (归一化)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 归一化所有指标到[0,1]
    df_norm = df.copy()
    for col in ['cd', 'ate', 'are', 'time_ms']:
        min_val = df[col].min()
        max_val = df[col].max()
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    
    # 绘制多指标雷达图替代方案：堆积柱状图
    x = np.arange(len(df))
    width = 0.6
    
    # 创建新的多目标优化指标: 加权和
    # 权重: CD(精度, 40%) + ATE(轨迹40%) + ARE(旋转10%) + Time(10%)
    weights = {'cd': 0.4, 'ate': 0.4, 'are': 0.1, 'time_ms': 0.1}
    overall_score = (df_norm['cd'] * weights['cd'] + 
                    df_norm['ate'] * weights['ate'] + 
                    df_norm['are'] * weights['are'] + 
                    df_norm['time_ms'] * weights['time_ms'])
    
    colors_score = ['#51CF66' if x < overall_score.min() + 0.1 else '#FFA94D' for x in overall_score]
    
    bars = ax.bar(x, overall_score, width, label='综合性能评分', color=colors_score, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, score in zip(bars, overall_score):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('综合性能评分 (越低越好)', fontsize=12, fontweight='bold')
    ax.set_xlabel('分区配置', fontsize=12, fontweight='bold')
    ax.set_title('多目标优化: 综合性能评分\n(权重: CD 40% + ATE 40% + ARE 10% + Time 10%)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dst']:.0%}" for _, row in df.iterrows()], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.text(0.5, 0.01, 'Dst分区比率 (%)', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_dir / 'overall_performance_score.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: overall_performance_score.png")
    plt.close()
    
    return overall_score


def generate_analysis_report(df: pd.DataFrame, overall_score, output_dir: Path):
    """生成详细的分析报告。"""
    
    baseline_idx = 0
    baseline_row = df.iloc[baseline_idx]
    best_idx = df['cd'].idxmin()
    best_row = df.iloc[best_idx]
    
    report = []
    report.append("="*80)
    report.append("Norm分区比率优化测试 - 详细分析报告")
    report.append("="*80)
    report.append("")
    
    # 基本信息
    report.append("【测试概述】")
    report.append(f"  测试场景: ScanNet数据集")
    report.append(f"  总配置数: {len(df)}")
    report.append(f"  所有配置均成功: {df['success'].all()}")
    report.append("")
    
    # 配置说明
    report.append("【分区配置说明】")
    report.append("  Protected (P): 保护分区 - 完全保留，不参与合并")
    report.append("  Dst (D): 目标分区 - Token合并的接收方")
    report.append("  Src (S): 源分区 - Token合并的提供方")
    report.append("  压缩率 = Src / Dst (数值越大，压缩越强)")
    report.append("")
    
    # 详细指标说明
    report.append("【性能指标说明】")
    report.append("  CD (Chamfer距离): 重建点云与真值的距离,越小越好,单位厘米")
    report.append("  ATE (累积循环误差): 轨迹估计误差,越小越好,单位米")
    report.append("  ARE (平均相对旋转误差): 姿态估计误差,越小越好,单位度")
    report.append("  Time (推理时间): 平均推理耗时,越短越好,单位毫秒")
    report.append("")
    
    # 基线分析
    report.append("【基线配置】")
    report.append(f"  配置: P={baseline_row['protected']:.0%} / D={baseline_row['dst']:.0%} / S={baseline_row['src']:.0%}")
    report.append(f"  CD: {baseline_row['cd']:.4f} cm")
    report.append(f"  ATE: {baseline_row['ate']:.4f} m")
    report.append(f"  ARE: {baseline_row['are']:.4f}°")
    report.append(f"  Time: {baseline_row['time_ms']:.1f} ms")
    report.append("")
    
    # 最优配置分析
    report.append("【最优配置(精度优先)】")
    report.append(f"  配置: P={best_row['protected']:.0%} / D={best_row['dst']:.0%} / S={best_row['src']:.0%}")
    report.append(f"  CD: {best_row['cd']:.4f} cm (改进 {(baseline_row['cd']-best_row['cd'])/baseline_row['cd']*100:.1f}%)")
    report.append(f"  ATE: {best_row['ate']:.4f} m (变化 {(best_row['ate']-baseline_row['ate'])/baseline_row['ate']*100:+.1f}%)")
    report.append(f"  ARE: {best_row['are']:.4f}° (改进 {(baseline_row['are']-best_row['are'])/baseline_row['are']*100:.1f}%)")
    report.append(f"  Time: {best_row['time_ms']:.1f} ms (变化 {(best_row['time_ms']-baseline_row['time_ms'])/baseline_row['time_ms']*100:+.1f}%)")
    report.append(f"  压缩率: {best_row['compression']:.2f}x")
    report.append("")
    
    # 综合评分分析
    report.append("【综合性能评分】")
    report.append("  (权重: CD 40% + ATE 40% + ARE 10% + Time 10%)")
    best_score_idx = overall_score.idxmin()
    report.append(f"  最优综合配置: D={df.iloc[best_score_idx]['dst']:.0%}")
    report.append(f"  综合评分: {overall_score.iloc[best_score_idx]:.4f}")
    report.append("")
    
    # 详细对比表
    report.append("【详细对比表】")
    report.append("-" * 100)
    report.append(f"{'配置':<15} {'P':<6} {'D':<6} {'S':<6} {'压缩率':<10} {'CD(cm)':<10} {'纠正(%)':<10} {'ATE(m)':<10} {'ARE(°)':<10} {'Time(ms)':<10}")
    report.append("-" * 100)
    
    for idx, row in df.iterrows():
        cd_improvement = (baseline_row['cd'] - row['cd']) / baseline_row['cd'] * 100
        report.append(f"{row['ratio']:<15} {row['protected']:.0%}{'':<4} {row['dst']:.0%}{'':<4} {row['src']:.0%}{'':<4} "
                     f"{row['compression']:>5.2f}x{'':<4} {row['cd']:>6.4f}{'':<3} {cd_improvement:>6.1f}%{'':<3} "
                     f"{row['ate']:>6.4f}{'':<3} {row['are']:>6.2f}{'':<3} {row['time_ms']:>8.1f}{'':<3}")
    
    report.append("-" * 100)
    report.append("")
    
    # 关键发现
    report.append("【关键发现】")
    
    # CD趋势分析
    cd_values = df['cd'].tolist()
    cd_trend = "单调递减" if all(cd_values[i] >= cd_values[i+1] for i in range(len(cd_values)-1)) else "非单调"
    report.append(f"  1. CD趋势: {cd_trend} - 增加Dst比率能够改进几何精度")
    report.append(f"     • 0.10→0.40的Dst提升: {(baseline_row['cd']-best_row['cd'])/baseline_row['cd']*100:.1f}% CD改进")
    
    # ATE-CD权衡
    ate_best_idx = df['ate'].idxmin()
    report.append(f"  2. ATE-CD权衡:")
    report.append(f"     • CD最优配置 vs ATE最优配置 ATE差异: {abs(df.iloc[ate_best_idx]['ate']-best_row['ate']):.4f} m")
    report.append(f"       (相差 {abs(df.iloc[ate_best_idx]['ate']-best_row['ate'])/df.iloc[ate_best_idx]['ate']*100:.1f}%)")
    
    # 时间成本
    time_increase = (best_row['time_ms'] - baseline_row['time_ms']) / baseline_row['time_ms'] * 100
    report.append(f"  3. 时间成本: 达到最优精度需要额外 {time_increase:.1f}% 推理时间")
    
    # 压缩效率
    report.append(f"  4. 压缩效率体现:")
    report.append(f"     • 基线压缩率: {baseline_row['compression']:.1f}x (高压缩,低精度)")
    report.append(f"     • 最优配置压缩率: {best_row['compression']:.1f}x (精度优先)")
    report.append("")
    
    # 推荐方案
    report.append("【推荐方案】")
    report.append(f"  ★ 生产环境推荐: P={best_row['protected']:.0%} / D={best_row['dst']:.0%} / S={best_row['src']:.0%}")
    report.append(f"    原因:")
    report.append(f"      • 几何精度最优: CD={best_row['cd']:.4f} cm (相对基线改进 {(baseline_row['cd']-best_row['cd'])/baseline_row['cd']*100:.1f}%)")
    report.append(f"      • 轨迹估计可靠: ATE={best_row['ate']:.4f} m")
    report.append(f"      • 旋转精度稳定: ARE={best_row['are']:.4f}°")
    report.append(f"      • 推理可接受: {best_row['time_ms']:.1f} ms")
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # 保存为文本文件
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ 分析报告已保存: {report_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Norm分区比率优化结果可视化')
    parser.add_argument('--input_csv', type=str,
                       default='tests/tests_result/norm_partition_optimization/partition_ratio_results.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='tests/tests_result/norm_partition_optimization',
                       help='输出目录，保存图表和分析报告')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 加载结果
    print("📊 加载测试结果...")
    df = load_results(args.input_csv)
    print(f"   ✓ 成功加载 {len(df)} 个配置的结果")
    print()
    
    # 创建可视化
    print("📈 生成可视化图表...")
    overall_score = create_visualizations(df, output_dir)
    print()
    
    # 生成分析报告
    print("📝 生成分析报告...")
    report_text = generate_analysis_report(df, overall_score, output_dir)
    print()
    
    # 打印报告到控制台
    print(report_text)
    
    print("\n✅ 所有可视化和分析完成！")
    print(f"   输出目录: {output_dir}")


if __name__ == "__main__":
    main()
