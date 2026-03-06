#!/usr/bin/env python3
"""
SOTA对标可视化脚本（生成论文级图表）
=====================================

生成6张论文级可视化图表，用于支撑"VGGT-Fast"在SOTA上的优势论证
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10
})
sns.set_style("whitegrid")
sns.set_palette("husl")


class SOTAVisualizer:
    def __init__(self, csv_dir: str = '/home/hba/Documents/FastVGGT/tests/tests_result/sota_comparison',
                 output_dir: str = None):
        self.csv_dir = Path(csv_dir)
        self.output_dir = Path(output_dir or csv_dir) / 'figures'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载数据
        self.df_raw = pd.read_csv(self.csv_dir / 'sota_comparison_raw.csv')
        self.df_summary = pd.read_csv(self.csv_dir / 'sota_comparison_summary.csv')
        
        # 数据类型转换
        for col in ['AUC@30', 'AUC@30_Std', 'CD_mm', 'CD_Std', 'Time_s', 'Time_Std']:
            self.df_summary[col] = pd.to_numeric(self.df_summary[col], errors='coerce')
        
        # 定义颜色方案
        self.color_map = {
            'Traditional': '#2ca02c',
            'DL': '#ff7f0e',
            'Ours': '#d62728',
            'Ours-Baseline': '#1f77b4'
        }
    
    def _get_colors(self, models):
        """获取模型对应的颜色"""
        colors = []
        for m in models:
            if 'VGGT-Fast' in m:
                colors.append(self.color_map['Ours'])
            elif 'VGGT' in m:
                colors.append(self.color_map['Ours-Baseline'])
            elif m in ['DUSt3R', 'MASt3R']:
                colors.append(self.color_map['DL'])
            else:  # COLMAP, VGGSfM
                colors.append(self.color_map['Traditional'])
        return colors
    
    def plot_1_accuracy_comparison(self):
        """图1：精度对比（条形图）"""
        fig, ax = plt.subplots(figsize=(13, 7))
        
        # 数据准备
        df_plot = self.df_summary.copy()
        df_plot = df_plot.sort_values('AUC@30', ascending=False)
        
        colors = self._get_colors(df_plot['Model'].tolist())
        
        # 绘制条形图
        x_pos = np.arange(len(df_plot))
        bars = ax.bar(x_pos, df_plot['AUC@30'], 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加误差棒
        ax.errorbar(x_pos, df_plot['AUC@30'], yerr=df_plot['AUC@30_Std'],
                   fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.5)
        
        # 标签和标题
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('AUC@30° (越高越好)', fontsize=12, fontweight='bold')
        ax.set_title('图1a：位姿估计精度对比（100帧长序列）', fontsize=13, fontweight='bold')
        ax.set_ylim([0.75, 0.95])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar, val, std in zip(bars, df_plot['AUC@30'], df_plot['AUC@30_Std']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 添加理想值参考线
        ax.axhline(y=0.92, color='green', linestyle='--', linewidth=2, alpha=0.5, label='COLMAP基准')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 图1: 01_accuracy_comparison.png")
        plt.close()
    
    def plot_2_efficiency_pareto(self):
        """图2：精度-速度Pareto前沿（散点图）"""
        fig, ax = plt.subplots(figsize=(13, 9))
        
        df_plot = self.df_summary.copy()
        
        colors = self._get_colors(df_plot['Model'].tolist())
        sizes = [400 if 'Ours' in m else 250 for m in df_plot['Model']]
        
        # 绘制散点
        scatter = ax.scatter(df_plot['Time_s'], df_plot['AUC@30'], 
                            s=sizes, c=colors, alpha=0.6, 
                            edgecolors='black', linewidth=2)
        
        # 标注每个点
        for idx, row in df_plot.iterrows():
            ax.annotate(row['Model'], 
                       (row['Time_s'], row['AUC@30']),
                       fontsize=9, fontweight='bold', 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 添加Pareto前沿标记（理想方向）
        ax.annotate('', xy=(25, 0.92), xytext=(400, 0.79),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.5))
        ax.text(200, 0.88, 'Pareto前沿方向\n(更快+更精准)', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # 标签和标题
        ax.set_xlabel('端到端推理耗时 (秒)', fontsize=12, fontweight='bold')
        ax.set_ylabel('位姿估计精度 AUC@30°', fontsize=12, fontweight='bold')
        ax.set_title('图1b：精度-速度权衡分析（长序列100帧）\n★VGGT-Fast实现"最快+最精准"的罕见组合', 
                    fontsize=13, fontweight='bold', color='darkred')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 500])
        ax.set_ylim([0.77, 0.94])
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='传统几何方法 (COLMAP, VGGSfM)', alpha=0.7),
            Patch(facecolor='#ff7f0e', label='深度学习方法 (DUSt3R, MASt3R)', alpha=0.7),
            Patch(facecolor='#d62728', label='本文方法 VGGT-Fast (Ours)', alpha=0.7),
            Patch(facecolor='#1f77b4', label='本文基线 VGGT (Original)', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_efficiency_pareto.png', dpi=300, bbox_inches='tight')
        print("✓ 图2: 02_efficiency_pareto.png")
        plt.close()
    
    def plot_3_timing_comparison(self):
        """图3：推理耗时对比（线性 + 对数坐标）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        df_plot = self.df_summary.copy()
        df_plot = df_plot.sort_values('Time_s', ascending=True)
        
        colors = self._get_colors(df_plot['Model'].tolist())
        
        # 线性坐标
        y_pos = np.arange(len(df_plot))
        bars1 = ax1.barh(y_pos, df_plot['Time_s'], 
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_plot['Model'], fontsize=10)
        ax1.set_xlabel('推理耗时 (秒)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) 线性尺度', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, df_plot['Time_s'])):
            ax1.text(val + 10, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}s', va='center', fontweight='bold', fontsize=9)
        
        # 添加加速比标注（相对原始VGGT）
        vggt_orig_time = df_plot[df_plot['Model'] == 'VGGT Original']['Time_s'].values[0]
        for i, (bar, val, model) in enumerate(zip(bars1, df_plot['Time_s'], df_plot['Model'])):
            if 'VGGT-Fast' in model:
                speedup = vggt_orig_time / val
                ax1.text(val + 60, bar.get_y() + bar.get_height()/2,
                        f'({speedup:.1f}x faster)', va='center', 
                        fontweight='bold', fontsize=9, color='darkred')
        
        # 对数坐标
        bars2 = ax2.barh(y_pos, df_plot['Time_s'],
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df_plot['Model'], fontsize=10)
        ax2.set_xlabel('推理耗时 (秒，对数坐标)', fontsize=11, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_title('(b) 对数尺度（强调快速方法间差异）', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        fig.suptitle('图2：端到端推理耗时对比（100帧长序列）', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_timing_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 图3: 03_timing_comparison.png")
        plt.close()
    
    def plot_4_reconstruction_quality(self):
        """图4：三维重建质量对比（倒角距离）"""
        fig, ax = plt.subplots(figsize=(13, 7))
        
        df_plot = self.df_summary.copy()
        df_plot = df_plot.sort_values('CD_mm', ascending=True)
        
        colors = self._get_colors(df_plot['Model'].tolist())
        
        x_pos = np.arange(len(df_plot))
        bars = ax.bar(x_pos, df_plot['CD_mm'],
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加误差棒
        ax.errorbar(x_pos, df_plot['CD_mm'], yerr=df_plot['CD_Std'],
                   fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('倒角距离 (mm，越低越好)', fontsize=12, fontweight='bold')
        ax.set_title('图3：三维重建质量对比（倒角距离）', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar, val in zip(bars, df_plot['CD_mm']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 添加参考线
        colmap_cd = df_plot[df_plot['Model'] == 'COLMAP']['CD_mm'].values[0]
        ax.axhline(y=colmap_cd, color='green', linestyle='--', linewidth=2, 
                  alpha=0.5, label='COLMAP基准')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_reconstruction_quality.png', dpi=300, bbox_inches='tight')
        print("✓ 图4: 04_reconstruction_quality.png")
        plt.close()
    
    def plot_5_composite_ranking(self):
        """图5：综合性能排名（雷达图）"""
        try:
            from math import pi
            
            df_plot = self.df_summary.copy()
            
            # 标准化指标到0-100分
            auc_norm = (df_plot['AUC@30'] / df_plot['AUC@30'].max()) * 100
            time_norm = (df_plot['Time_s'].max() / df_plot['Time_s']) * 100  # 倒数
            cd_norm = (df_plot['CD_mm'].max() / df_plot['CD_mm']) * 100  # 倒数
            
            # 选择关键模型进行对比
            key_models = ['COLMAP', 'MASt3R', 'VGGT Original', 'VGGT-Fast (Ours)']
            df_radar = df_plot[df_plot['Model'].isin(key_models)].copy()
            
            categories = ['精度\nAUC@30', '速度\n(推理时间)', '重建质量\n(倒角距离)']
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            color_radar = {'COLMAP': '#2ca02c', 'MASt3R': '#ff7f0e',
                          'VGGT Original': '#1f77b4', 'VGGT-Fast (Ours)': '#d62728'}
            
            for idx, row in df_radar.iterrows():
                model = row['Model']
                idx_full = df_plot[df_plot['Model'] == model].index[0]
                
                values = [auc_norm[idx_full], time_norm[idx_full], cd_norm[idx_full]]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2.5, label=model,
                       color=color_radar[model], markersize=8)
                ax.fill(angles, values, alpha=0.15, color=color_radar[model])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
            ax.grid(True, linestyle='--')
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), 
                      fontsize=10, framealpha=0.95)
            plt.title('图4：综合性能评估（标准化得分：0-100）', 
                     fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / '05_composite_radar.png', dpi=300, bbox_inches='tight')
            print("✓ 图5: 05_composite_radar.png")
            plt.close()
        except Exception as e:
            print(f"⚠ 图5生成失败: {e}")
    
    def plot_6_dataset_breakdown(self):
        """图6：数据集泛化性对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        for ax, dataset in zip([ax1, ax2], ['7-Scenes', 'ScanNet']):
            df_ds = self.df_raw[self.df_raw['dataset'] == dataset].groupby('model').agg({
                'auc@30': ['mean', 'std'],
                'overall_cd': 'mean',
                'inference_time_s': 'mean'
            }).reset_index()
            
            df_ds.columns = ['model', 'auc_mean', 'auc_std', 'cd_mean', 'time_mean']
            df_ds = df_ds.sort_values('auc_mean', ascending=False)
            
            colors = self._get_colors(df_ds['model'].tolist())
            
            x_pos = np.arange(len(df_ds))
            bars = ax.bar(x_pos, df_ds['auc_mean'],
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.errorbar(x_pos, df_ds['auc_mean'], yerr=df_ds['auc_std'],
                       fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_ds['model'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('AUC@30°', fontsize=11, fontweight='bold')
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_ylim([0.75, 0.95])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, val in zip(bars, df_ds['auc_mean']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        fig.suptitle('图5：跨数据集泛化性能对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_dataset_breakdown.png', dpi=300, bbox_inches='tight')
        print("✓ 图6: 06_dataset_breakdown.png")
        plt.close()
    
    def generate_all(self):
        """生成所有论文级图表"""
        print("\n【生成SOTA对标论文级图表（6张）】\n")
        self.plot_1_accuracy_comparison()
        self.plot_2_efficiency_pareto()
        self.plot_3_timing_comparison()
        self.plot_4_reconstruction_quality()
        self.plot_5_composite_ranking()
        self.plot_6_dataset_breakdown()
        print(f"\n✨ 所有图表已保存至: {self.output_dir}\n")


def main():
    viz = SOTAVisualizer()
    viz.generate_all()


if __name__ == '__main__':
    main()
