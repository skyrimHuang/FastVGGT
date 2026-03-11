#!/usr/bin/env python3
"""
绘制ScaleHead实验结果可视化图表

生成图3.14：(a) GT尺度 vs. 预测尺度散点图; (b) 尺度误差分布直方图
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager


def setup_chinese_font() -> None:
    """全局设置中文字体，避免乱码。"""
    cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    try:
        font_manager.fontManager.addfont(cjk_font_path)
    except Exception:
        pass

    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',
        'SimHei',
        'DejaVu Sans',
    ]
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 11


setup_chinese_font()

def plot_scalehead_results(csv_path: str, output_path: str):
    """
    生成ScaleHead实验结果可视化
    
    Args:
        csv_path: 详细结果CSV路径
        output_path: 输出图片路径
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建2×1子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ========== (a) GT尺度 vs. 预测尺度散点图 ==========
    ax = axes[0]
    
    gt_scale = df['gt_scale'].values
    pred_scale = df['pred_scale'].values
    scale_error = df['scale_error_pct'].values
    
    # 色彩编码：误差大小
    scatter = ax.scatter(
        gt_scale, pred_scale,
        c=scale_error,
        cmap='RdYlGn_r',  # 红色=高误差，绿色=低误差
        s=60,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    
    # 理想对角线 y=x
    min_scale = min(gt_scale.min(), pred_scale.min())
    max_scale = max(gt_scale.max(), pred_scale.max())
    ax.plot([min_scale, max_scale], [min_scale, max_scale], 
            'k--', linewidth=1.5, alpha=0.6, label='理想线 (y=x)')
    
    # ±15% error band
    x_range = np.linspace(min_scale, max_scale, 100)
    ax.fill_between(x_range, x_range * 0.85, x_range * 1.15,
                     color='gray', alpha=0.2, label='±15%误差带')
    
    ax.set_xlabel('真实尺度', fontsize=12)
    ax.set_ylabel('预测尺度', fontsize=12)
    ax.set_title('(a) 真实尺度与预测尺度对比', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('尺度误差 (%)', fontsize=11)
    
    # ========== (b) 尺度误差分布直方图 ==========
    ax = axes[1]
    
    # 绘制直方图
    n, bins, patches = ax.hist(
        scale_error,
        bins=15,
        range=(0, 80),
        color='steelblue',
        edgecolor='k',
        alpha=0.7,
        linewidth=0.8
    )
    
    # 标注统计信息
    mean_err = scale_error.mean()
    median_err = np.median(scale_error)
    std_err = scale_error.std()
    
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
               label=f'均值: {mean_err:.1f}%')
    ax.axvline(median_err, color='darkgreen', linestyle='-.', linewidth=2,
               label=f'中位数: {median_err:.1f}%')
    
    ax.set_xlabel('尺度误差 (%)', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('(b) 尺度误差分布 (N=50)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("尺度预测统计")
    print(f"{'='*60}")
    print(f"平均误差:     {mean_err:.2f}%")
    print(f"中位误差:     {median_err:.2f}%")
    print(f"标准差:       {std_err:.2f}%")
    print(f"最小误差:     {scale_error.min():.2f}%")
    print(f"最大误差:     {scale_error.max():.2f}%")
    print(f"误差 < 5% 样本:  {(scale_error < 5).sum()} / {len(scale_error)}")
    print(f"误差 < 10%样本: {(scale_error < 10).sum()} / {len(scale_error)}")
    print(f"误差 < 20%样本: {(scale_error < 20).sum()} / {len(scale_error)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="绘制ScaleHead评估结果图")
    parser.add_argument('--csv', type=str, required=True,
                        help="详细结果CSV路径")
    parser.add_argument('--output', type=str, default='scalehead_evaluation.png',
                        help="图像输出路径")
    args = parser.parse_args()
    
    plot_scalehead_results(args.csv, args.output)


if __name__ == "__main__":
    main()
