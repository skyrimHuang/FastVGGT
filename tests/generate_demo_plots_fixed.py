#!/usr/bin/env python3
"""
Generate demo plots with global Chinese font loading.
NOTICE: ILLUSTRATIVE DEMO DATA - NOT REAL EXPERIMENTAL RESULTS.

Plot order follows thesis narrative:
1) 普通场景性能
2) 大基线场景性能
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')


def setup_chinese_font() -> str:
    """Globally set a Chinese-capable font for matplotlib."""
    # 清理缓存，避免旧字体映射
    cache_dir = os.path.expanduser('~/.cache/matplotlib')
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            if name.startswith('fontlist-'):
                try:
                    os.remove(os.path.join(cache_dir, name))
                except OSError:
                    pass

    candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'AR PL UMing CN',
        'AR PL UKai CN',
        'WenQuanYi Micro Hei',
        'SimHei',
        'Microsoft YaHei',
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    selected = None
    for c in candidates:
        if c in available:
            selected = c
            break

    if selected is None:
        # 部分系统会只有 CJK JP
        partial = [f for f in available if 'CJK' in f or 'AR PL' in f]
        selected = partial[0] if partial else 'DejaVu Sans'

    plt.rcParams['font.sans-serif'] = [selected, 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return selected


def get_regime_all(df_summary: pd.DataFrame, regime: str) -> pd.DataFrame:
    out = df_summary[(df_summary['regime'] == regime) & (df_summary['scene'] == 'ALL')].copy()
    order = {'A': 0, 'B': 1, 'C': 2}
    out['__o'] = out['method'].map(order)
    out = out.sort_values('__o').drop(columns='__o')
    return out


METHOD_LABELS = {
    'A': '几何管线',
    'B': '语义管线',
    'C': '混合管线',
}


def main():
    font_name = setup_chinese_font()
    print(f"[PLOT] Chinese font selected: {font_name}")

    workspace = Path('/home/hba/Documents/FastVGGT')
    demo_dir = workspace / 'tests/tests_result/hybrid_registration_7scenes/demo'
    pair_csv = demo_dir / 'hybrid_registration_pair_results_demo.csv'
    summary_csv = demo_dir / 'hybrid_registration_summary_demo.csv'

    df_pair = pd.read_csv(pair_csv)
    df_summary = pd.read_csv(summary_csv)

    if 'data_type' in df_pair.columns:
        df_pair = df_pair.drop(columns=['data_type'])

    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    # ---------- Figure 1: 普通场景 ----------
    n_all = get_regime_all(df_summary, '普通场景')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('普通场景下的方法性能对比', fontweight='bold')

    metrics = [
        ('mean_rotation_error_deg', '旋转误差 (度)'),
        ('mean_translation_error_m', '平移误差 (米)'),
        ('mean_chamfer_distance', 'Chamfer距离'),
    ]

    for ax, (col, title) in zip(axes, metrics):
        vals = n_all[col].values
        methods = n_all['method'].values
        bars = ax.bar(methods, vals, color=[colors[m] for m in methods], edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=0)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.2)
        for i, b in enumerate(bars):
            fmt = '{:.3f}' if 'translation' in col or 'chamfer' in col else '{:.2f}'
            suffix = '°' if 'rotation' in col else ''
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(vals[i]) + suffix,
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig1 = demo_dir / 'demo_figure1_method_comparison.png'
    plt.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {fig1}")

    # ---------- Figure 2: 大基线场景 ----------
    l_all = get_regime_all(df_summary, '大基线场景')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('大基线场景下的方法性能对比', fontweight='bold')

    for ax, (col, title) in zip(axes, metrics):
        vals = l_all[col].values
        methods = l_all['method'].values
        bars = ax.bar(methods, vals, color=[colors[m] for m in methods], edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=0)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.2)
        for i, b in enumerate(bars):
            fmt = '{:.3f}' if 'translation' in col or 'chamfer' in col else '{:.2f}'
            suffix = '°' if 'rotation' in col else ''
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(vals[i]) + suffix,
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig2 = demo_dir / 'demo_figure2_scene_distribution.png'
    plt.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {fig2}")

    # ---------- Figure 3: 旋转-平移散点（分 regime） ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('误差相关性对比：普通场景 vs 大基线场景', fontweight='bold')

    for ax, regime in zip(axes, ['普通场景', '大基线场景']):
        sub = df_pair[df_pair['regime'] == regime]
        for method in ['A', 'B', 'C']:
            m = sub[sub['method'] == method]
            ax.scatter(
                m['rotation_error_deg'],
                m['translation_error_m'],
                s=36,
                alpha=0.65,
                color=colors[method],
                edgecolors='black',
                linewidth=0.3,
                label=METHOD_LABELS[method]
            )
        ax.set_title(regime, fontweight='bold')
        ax.set_xlabel('旋转误差 (度)')
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel('平移误差 (米)')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig3 = demo_dir / 'demo_figure3_error_correlation.png'
    plt.savefig(fig3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {fig3}")

    # ---------- Figure 4: C 相对 B 的改进率（双 regime） ----------
    fig, ax = plt.subplots(figsize=(9, 5))

    regimes = ['普通场景', '大基线场景']
    metrics_label = ['旋转误差', '平移误差', 'Chamfer距离']

    def imp(regime, col):
        sub = get_regime_all(df_summary, regime)
        b = float(sub[sub['method'] == 'B'][col].iloc[0])
        c = float(sub[sub['method'] == 'C'][col].iloc[0])
        return (1.0 - c / b) * 100.0

    normal_imp = [
        imp('普通场景', 'mean_rotation_error_deg'),
        imp('普通场景', 'mean_translation_error_m'),
        imp('普通场景', 'mean_chamfer_distance'),
    ]
    large_imp = [
        imp('大基线场景', 'mean_rotation_error_deg'),
        imp('大基线场景', 'mean_translation_error_m'),
        imp('大基线场景', 'mean_chamfer_distance'),
    ]

    x = np.arange(len(metrics_label))
    w = 0.34
    b1 = ax.bar(x - w / 2, normal_imp, w, label='普通场景', color='#4c78a8')
    b2 = ax.bar(x + w / 2, large_imp, w, label='大基线场景', color='#f58518')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_label)
    ax.set_ylabel('改进率（%）')
    ax.set_title('混合管线相对语义管线的改进率对比', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig4 = demo_dir / 'demo_figure4_multimet_comparison.png'
    plt.savefig(fig4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {fig4}")

    # ---------- Figure 5: 耗时对比（参考 scene summary 风格） ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('方法耗时对比：普通场景 vs 大基线场景', fontweight='bold')

    for ax, regime in zip(axes, ['普通场景', '大基线场景']):
        sub = df_summary[(df_summary['regime'] == regime) & (df_summary['scene'].isin(['office', 'redkitchen']))].copy()
        scene_order = ['office', 'redkitchen']
        method_order = ['A', 'B', 'C']

        x = np.arange(len(scene_order))
        w = 0.24

        for i, method in enumerate(method_order):
            vals = []
            for scene in scene_order:
                row = sub[(sub['scene'] == scene) & (sub['method'] == method)]
                vals.append(float(row['mean_runtime_ms'].iloc[0]))

            bars = ax.bar(
                x + (i - 1) * w,
                vals,
                width=w,
                label=METHOD_LABELS[method],
                color=colors[method],
                edgecolor='black',
                alpha=0.85,
            )
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h, f'{h:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(regime, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Office', 'RedKitchen'])
        ax.set_xlabel('场景')
        ax.grid(axis='y', alpha=0.25)

    axes[0].set_ylabel('平均耗时 (ms)')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig5 = demo_dir / 'demo_figure5_runtime_scene_summary.png'
    plt.savefig(fig5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {fig5}")

    print('[PLOT] Done. (DEMO data only)')


if __name__ == '__main__':
    main()
