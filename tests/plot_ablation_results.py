#!/usr/bin/env python3
"""
时空自适应合并策略消融实验可视化脚本
=====================================

生成论文级别的图表，展示Full-Adaptive策略相对其他方法的优势。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict
import argparse


def setup_chinese_font() -> None:
    """全局设置中文字体，采用动态扫描方案"""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    
    # 扫描系统已安装的字体
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    
    # 从preferred_fonts中筛选已安装的字体
    selected_fonts = [name for name in preferred_fonts if name in available_font_names]
    
    if not selected_fonts:
        selected_fonts = preferred_fonts
    
    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


class AblationVisualizer:
    """消融实验可视化器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.df = pd.read_csv(self.data_dir / "ablation_spatiotemporal_adaptive_raw.csv")
        self.summary_df = pd.read_csv(self.data_dir / "ablation_summary_statistics.csv")
        
        # 策略顺序和颜色映射
        self.strategy_order = ['Baseline', 'Fixed-20%', 'Fixed-40%', 'Adaptive-Layer', 'Full-Adaptive']
        self.strategy_colors = {
            'Baseline': '#95a5a6',          # 灰色
            'Fixed-20%': '#3498db',         # 蓝色
            'Fixed-40%': '#9b59b6',         # 紫色
            'Adaptive-Layer': '#e67e22',    # 橙色
            'Full-Adaptive': '#e74c3c',     # 红色（强调）
        }
        self.strategy_markers = {
            'Baseline': 'o',
            'Fixed-20%': 's',
            'Fixed-40%': '^',
            'Adaptive-Layer': 'D',
            'Full-Adaptive': '*',
        }
        self.strategy_linestyles = {
            'Baseline': '--',
            'Fixed-20%': '-.',
            'Fixed-40%': ':',
            'Adaptive-Layer': '-',
            'Full-Adaptive': '-',
        }
        
        # 中文策略名称映射
        self.strategy_cn_names = {
            'Baseline': '原始VGGT\n(无合并)',
            'Fixed-20%': '固定合并\n(20%)',
            'Fixed-40%': '固定合并\n(40%)',
            'Adaptive-Layer': '层级自适应\n(仅C_base)',
            'Full-Adaptive': '完整时空自适应\n(Ours)',
        }
        
        # 设置中文字体
        setup_chinese_font()
    
    def plot_ate_vs_seq_length(self, dataset: str = 'scannet'):
        """
        图1: ATE vs 序列长度
        展示Full-Adaptive在长序列下的精度优势
        """
        df_dataset = self.df[self.df['dataset'] == dataset]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in self.strategy_order:
            df_strategy = df_dataset[df_dataset['strategy'] == strategy]
            
            # 分离OOM和非OOM的数据点
            df_normal = df_strategy[df_strategy['is_oom'] == False]  # 非OOM点
            df_oom = df_strategy[df_strategy['is_oom'] == True]  # OOM点
            
            # 绘制正常点的曲线
            if len(df_normal) > 0:
                ax.plot(
                    df_normal['seq_length'], 
                    df_normal['ate'],
                    label=self.strategy_cn_names[strategy],
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle=self.strategy_linestyles[strategy],
                    linewidth=2.5 if strategy == 'Full-Adaptive' else 1.8,
                    markersize=10 if strategy == 'Full-Adaptive' else 7,
                    alpha=0.9
                )
            
            # 只在折线末端标记红色X（最后一个正常点）
            if len(df_oom) > 0 and len(df_normal) > 0:
                # 获取最后一个正常点的值
                last_normal_ate = df_normal['ate'].iloc[-1]
                last_normal_seq_len = df_normal['seq_length'].iloc[-1]
                
                # 在折线末端标记红色X
                ax.scatter(
                    [last_normal_seq_len],
                    [last_normal_ate],
                    marker='x',
                    color='red',
                    s=250,
                    linewidths=3,
                    zorder=15,
                    alpha=0.9
                )
        
        ax.set_xlabel('序列长度 (帧)', fontsize=14, fontweight='bold')
        ax.set_ylabel('绝对轨迹误差 ATE (m)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'消融实验：定位精度(ATE) vs 序列长度 [{dataset.upper()}]',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(5, 155)
        
        plt.tight_layout()
        output_file = self.output_dir / f"ablation_ate_vs_seqlen_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图1: {output_file}")
    
    def plot_vram_vs_seq_length(self, dataset: str = 'scannet'):
        """
        图2: 显存占用 vs 序列长度
        展示Full-Adaptive的显存节省效果，用红色X标记第一个OOM点
        """
        df_dataset = self.df[self.df['dataset'] == dataset]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        oom_threshold_gb = 21  # 21GB显存上限
        
        for strategy in self.strategy_order:
            df_strategy = df_dataset[df_dataset['strategy'] == strategy]
            
            # 分离OOM和非OOM的数据点
            df_normal = df_strategy[df_strategy['is_oom'] == False]  # 非OOM点
            df_oom = df_strategy[df_strategy['is_oom'] == True]  # OOM点
            
            # 绘制正常点的曲线
            if len(df_normal) > 0:
                ax.plot(
                    df_normal['seq_length'], 
                    df_normal['vram_mb'] / 1024,  # 转换为GB
                    label=self.strategy_cn_names[strategy],
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle=self.strategy_linestyles[strategy],
                    linewidth=2.5 if strategy == 'Full-Adaptive' else 1.8,
                    markersize=10 if strategy == 'Full-Adaptive' else 7,
                    alpha=0.9
                )
            
            # 只在第一个OOM点处标记红色X，并连接折线
            if len(df_oom) > 0 and len(df_normal) > 0:
                # 获取第一个OOM点的序列长度
                first_oom_seq_len = df_oom['seq_length'].iloc[0]
                
                # 获取最后一个正常点的显存值
                last_normal_vram_gb = df_normal['vram_mb'].iloc[-1] / 1024
                last_normal_seq_len = df_normal['seq_length'].iloc[-1]
                
                # 从最后一个正常点连线到第一个OOM点（在21GB处）
                ax.plot(
                    [last_normal_seq_len, first_oom_seq_len],
                    [last_normal_vram_gb, oom_threshold_gb],
                    color=self.strategy_colors[strategy],
                    linestyle=self.strategy_linestyles[strategy],
                    linewidth=2.5 if strategy == 'Full-Adaptive' else 1.8,
                    alpha=0.9
                )
                
                # 在第一个OOM点处标记红色X
                ax.scatter(
                    [first_oom_seq_len],
                    [oom_threshold_gb],
                    marker='x',
                    color='red',
                    s=250,
                    linewidths=3,
                    zorder=15,
                    alpha=0.9
                )
        
        # 添加OOM红线标注（21GB显存上限）
        ax.axhline(y=oom_threshold_gb, color='red', linestyle='--', linewidth=2.5, alpha=0.75, zorder=10)
        ax.text(135, oom_threshold_gb + 0.8, 'GPU显存上限\n(21GB OOM)', fontsize=11, 
                color='red', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9))
        
        ax.set_xlabel('序列长度 (帧)', fontsize=14, fontweight='bold')
        ax.set_ylabel('显存占用 (GB)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'消融实验：显存占用 vs 序列长度 [{dataset.upper()}]',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(5, 155)
        ax.set_ylim(0, 25)  # 留出OOM线空间
        
        plt.tight_layout()
        output_file = self.output_dir / f"ablation_vram_vs_seqlen_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图2: {output_file}")
    
    def plot_time_vs_seq_length(self, dataset: str = 'scannet'):
        """
        图3: 推理时间 vs 序列长度
        展示Full-Adaptive的加速效果
        """
        df_dataset = self.df[self.df['dataset'] == dataset]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in self.strategy_order:
            df_strategy = df_dataset[df_dataset['strategy'] == strategy]
            
            # 分离OOM和非OOM的数据点
            df_normal = df_strategy[df_strategy['is_oom'] == False]  # 非OOM点
            df_oom = df_strategy[df_strategy['is_oom'] == True]  # OOM点
            
            # 绘制正常点的曲线
            if len(df_normal) > 0:
                ax.plot(
                    df_normal['seq_length'], 
                    df_normal['inference_time_ms'] / 1000,  # 转换为秒
                    label=self.strategy_cn_names[strategy],
                    color=self.strategy_colors[strategy],
                    marker=self.strategy_markers[strategy],
                    linestyle=self.strategy_linestyles[strategy],
                    linewidth=2.5 if strategy == 'Full-Adaptive' else 1.8,
                    markersize=10 if strategy == 'Full-Adaptive' else 7,
                    alpha=0.9
                )
            
            # 只在折线末端标记红色X（最后一个正常点）
            if len(df_oom) > 0 and len(df_normal) > 0:
                # 获取最后一个正常点的值
                last_normal_time = df_normal['inference_time_ms'].iloc[-1] / 1000  # 转换为秒
                last_normal_seq_len = df_normal['seq_length'].iloc[-1]
                
                # 在折线末端标记红色X
                ax.scatter(
                    [last_normal_seq_len],
                    [last_normal_time],
                    marker='x',
                    color='red',
                    s=250,
                    linewidths=3,
                    zorder=15,
                    alpha=0.9
                )
        
        ax.set_xlabel('序列长度 (帧)', fontsize=14, fontweight='bold')
        ax.set_ylabel('推理时间 (秒)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'消融实验：推理时间 vs 序列长度 [{dataset.upper()}]',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(5, 155)
        
        plt.tight_layout()
        output_file = self.output_dir / f"ablation_time_vs_seqlen_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图3: {output_file}")
    
    def plot_comprehensive_comparison(self):
        """
        图4: 综合对比图（论文主图）
        3x2子图网格，展示ScanNet和KITTI的三个指标对比
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        datasets = ['scannet', 'kitti']
        metrics = [
            ('ate', 'ATE (m)', 'ATE'),
            ('vram_mb', '显存 (GB)', lambda x: x/1024),
            ('inference_time_ms', '推理时间 (秒)', lambda x: x/1000)
        ]
        
        oom_threshold_gb = 21  # 21GB显存上限
        
        for col_idx, dataset in enumerate(datasets):
            df_dataset = self.df[self.df['dataset'] == dataset]
            
            for row_idx, (metric, ylabel, transform) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                for strategy in self.strategy_order:
                    df_strategy = df_dataset[df_dataset['strategy'] == strategy]
                    
                    # 对于VRAM，特别处理OOM点
                    if row_idx == 1:  # vram_mb
                        df_normal = df_strategy[df_strategy['is_oom'] == False]
                        df_oom = df_strategy[df_strategy['is_oom'] == True]
                        
                        # 绘制正常点
                        if len(df_normal) > 0:
                            y_data = df_normal[metric]
                            if callable(transform):
                                y_data = transform(y_data)
                            
                            ax.plot(
                                df_normal['seq_length'], 
                                y_data,
                                label=self.strategy_cn_names[strategy],
                                color=self.strategy_colors[strategy],
                                marker=self.strategy_markers[strategy],
                                linestyle=self.strategy_linestyles[strategy],
                                linewidth=2.5 if strategy == 'Full-Adaptive' else 1.5,
                                markersize=8 if strategy == 'Full-Adaptive' else 5,
                                alpha=0.9
                            )
                        
                        # 只在第一个OOM点处标记红色X，并连接折线
                        if len(df_oom) > 0 and len(df_normal) > 0:
                            # 获取第一个OOM点的序列长度
                            first_oom_seq_len = df_oom['seq_length'].iloc[0]
                            
                            # 获取最后一个正常点的显存值
                            last_normal_vram = df_normal[metric].iloc[-1]
                            if callable(transform):
                                last_normal_vram = transform(last_normal_vram)
                            last_normal_seq_len = df_normal['seq_length'].iloc[-1]
                            
                            # 从最后一个正常点连线到第一个OOM点（在21GB处）
                            ax.plot(
                                [last_normal_seq_len, first_oom_seq_len],
                                [last_normal_vram, oom_threshold_gb],
                                color=self.strategy_colors[strategy],
                                linestyle=self.strategy_linestyles[strategy],
                                linewidth=2.5 if strategy == 'Full-Adaptive' else 1.5,
                                alpha=0.9
                            )
                            
                            # 在第一个OOM点处标记红色X
                            ax.scatter(
                                [first_oom_seq_len],
                                [oom_threshold_gb],
                                marker='x',
                                color='red',
                                s=150,
                                linewidths=2.5,
                                zorder=15,
                                alpha=0.9
                            )
                    else:
                        # 非VRAM指标也需要处理OOM
                        df_normal = df_strategy[df_strategy['is_oom'] == False]
                        df_oom = df_strategy[df_strategy['is_oom'] == True]
                        
                        # 绘制正常点
                        if len(df_normal) > 0:
                            y_data = df_normal[metric]
                            if callable(transform):
                                y_data = transform(y_data)
                            
                            ax.plot(
                                df_normal['seq_length'], 
                                y_data,
                                label=self.strategy_cn_names[strategy],
                                color=self.strategy_colors[strategy],
                                marker=self.strategy_markers[strategy],
                                linestyle=self.strategy_linestyles[strategy],
                                linewidth=2.5 if strategy == 'Full-Adaptive' else 1.5,
                                markersize=8 if strategy == 'Full-Adaptive' else 5,
                                alpha=0.9
                            )
                        
                        # 只在折线末端标记红色X（最后一个正常点）
                        if len(df_oom) > 0 and len(df_normal) > 0:
                            last_normal_seq_len = df_normal['seq_length'].iloc[-1]
                            last_normal_value = df_normal[metric].iloc[-1]
                            if callable(transform):
                                last_normal_value = transform(last_normal_value)
                            
                            # 在折线末端标记红色X
                            ax.scatter(
                                [last_normal_seq_len],
                                [last_normal_value],
                                marker='x',
                                color='red',
                                s=150,
                                linewidths=2.5,
                                zorder=15,
                                alpha=0.9
                            )
                
                # 在VRAM子图中添加OOM红线
                if row_idx == 1:  # vram_mb
                    ax.axhline(y=oom_threshold_gb, color='red', linestyle='--', linewidth=1.5, alpha=0.6, zorder=10)
                    if col_idx == 1:  # 只在右侧KITTI图中标注
                        ax.text(140, oom_threshold_gb + 0.5, '21GB OOM', fontsize=9, color='red', ha='right', fontweight='bold')
                    ax.set_ylim(0, 25)
                
                ax.set_xlabel('序列长度 (帧)', fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(f'{dataset.upper()} - {metrics[row_idx][2] if isinstance(metrics[row_idx][2], str) else ylabel}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(5, 155)
                
                # 只在第一个子图显示图例
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
        
        fig.suptitle(
            '时空自适应合并策略消融实验 - 综合性能对比',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_file = self.output_dir / "ablation_comprehensive_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图4（论文主图）: {output_file}")
    
    def plot_efficiency_pareto(self, dataset: str = 'scannet', seq_length: int = 100):
        """
        图5: 效率-精度Pareto前沿
        展示Full-Adaptive在特定序列长度下的优势
        """
        df_subset = self.df[(self.df['dataset'] == dataset) & (self.df['seq_length'] == seq_length)]
        
        # 检查是否有足够的数据点
        if len(df_subset) < 2:
            print(f"⚠ 跳过Pareto图（{dataset}, {seq_length}帧）：数据点不足（可能多个策略已OOM）")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点
        for strategy in self.strategy_order:
            strategy_data = df_subset[df_subset['strategy'] == strategy]
            if len(strategy_data) == 0:
                continue  # 该策略在该序列长度已OOM，跳过
            
            row = strategy_data.iloc[0]
            
            ax.scatter(
                row['speedup_pct'],
                row['ate'],
                label=self.strategy_cn_names[strategy],
                color=self.strategy_colors[strategy],
                marker=self.strategy_markers[strategy],
                s=400 if strategy == 'Full-Adaptive' else 200,
                alpha=0.8,
                edgecolors='black',
                linewidths=2 if strategy == 'Full-Adaptive' else 1
            )
            
            # 添加策略名称标注
            ax.annotate(
                self.strategy_cn_names[strategy],
                (row['speedup_pct'], row['ate']),
                xytext=(10, 5 if strategy != 'Baseline' else -15),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold' if strategy == 'Full-Adaptive' else 'normal',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow' if strategy == 'Full-Adaptive' else 'white', alpha=0.7)
            )
        
        ax.set_xlabel('加速比 (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('ATE (m)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'效率-精度权衡 ({dataset.upper()}, 序列长度={seq_length}帧)',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 标注理想区域（右下角：高加速、低误差）
        if len(df_subset) > 0:
            ax.axhline(y=df_subset['ate'].median(), color='green', linestyle='--', alpha=0.3, linewidth=1.5)
            ax.axvline(x=df_subset['speedup_pct'].median(), color='green', linestyle='--', alpha=0.3, linewidth=1.5)
            ax.text(
                ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] * 1.05,
                '理想区域\n(高效+精确)',
                fontsize=11,
                color='green',
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3)
            )
        
        plt.tight_layout()
        output_file = self.output_dir / f"ablation_pareto_{dataset}_L{seq_length}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图5: {output_file}")
    
    def plot_bar_comparison_table(self):
        """
        图6: 柱状图对比表
        展示各策略在关键指标上的量化对比（平均值）
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        datasets = ['scannet', 'kitti']
        metrics = [
            ('ate_mean', 'ATE (m)', False),
            ('vram_mean_mb', '显存 (MB)', False),
            ('time_mean_ms', '推理时间 (ms)', False),
            ('vram_reduction_mean_pct', '显存节省率 (%)', True)  # 越大越好
        ]
        
        for idx, (metric, ylabel, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            width = 0.35
            x = np.arange(len(self.strategy_order))
            
            for dataset_idx, dataset in enumerate(datasets):
                df_dataset = self.summary_df[self.summary_df['dataset'] == dataset]
                
                values = []
                for strategy in self.strategy_order:
                    row = df_dataset[df_dataset['strategy'] == strategy]
                    values.append(row[metric].values[0])
                
                offset = width * (-0.5 + dataset_idx)
                bars = ax.bar(
                    x + offset, 
                    values, 
                    width, 
                    label=dataset.upper(),
                    alpha=0.8
                )
                
                # 在柱状图上标注数值
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height,
                        f'{height:.1f}',
                        ha='center', 
                        va='bottom',
                        fontsize=8
                    )
            
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'{ylabel}对比', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.strategy_order, rotation=15, ha='right', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        fig.suptitle(
            '消融实验 - 关键指标量化对比',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_file = self.output_dir / "ablation_bar_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 生成图6: {output_file}")
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("\n【生成消融实验图表】")
        print("=" * 80)
        
        # 为每个数据集生成基础对比图
        for dataset in ['scannet', 'kitti']:
            print(f"\n处理数据集: {dataset.upper()}")
            self.plot_ate_vs_seq_length(dataset)
            self.plot_vram_vs_seq_length(dataset)
            self.plot_time_vs_seq_length(dataset)
        
        # 生成综合对比图（论文主图）
        print("\n生成综合对比图...")
        self.plot_comprehensive_comparison()
        
        # 生成Pareto前沿图
        print("\n生成Pareto前沿图...")
        for dataset in ['scannet', 'kitti']:
            self.plot_efficiency_pareto(dataset, seq_length=100)
        
        # 生成柱状图对比
        print("\n生成柱状图对比...")
        self.plot_bar_comparison_table()
        
        print("\n" + "=" * 80)
        print("✨ 所有图表生成完成！")


def main():
    parser = argparse.ArgumentParser(description='生成消融实验可视化图表')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='tests/tests_result/ablation_spatiotemporal',
        help='数据输入目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tests/tests_result/ablation_spatiotemporal/figures',
        help='图表输出目录'
    )
    args = parser.parse_args()
    
    visualizer = AblationVisualizer(args.data_dir, args.output_dir)
    visualizer.generate_all_figures()
    
    print(f"\n输出目录: {args.output_dir}")
    print("\n生成的图表:")
    print("  【基础对比图】")
    print("    - ablation_ate_vs_seqlen_scannet.png")
    print("    - ablation_ate_vs_seqlen_kitti.png")
    print("    - ablation_vram_vs_seqlen_scannet.png")
    print("    - ablation_vram_vs_seqlen_kitti.png")
    print("    - ablation_time_vs_seqlen_scannet.png")
    print("    - ablation_time_vs_seqlen_kitti.png")
    print("  【综合分析图】")
    print("    - ablation_comprehensive_comparison.png  ★ 论文主图")
    print("    - ablation_pareto_scannet_L100.png")
    print("    - ablation_pareto_kitti_L100.png")
    print("    - ablation_bar_comparison.png")


if __name__ == "__main__":
    main()
