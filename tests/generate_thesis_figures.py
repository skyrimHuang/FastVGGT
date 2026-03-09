#!/usr/bin/env python3
"""
Generate figures for thesis section 3.4.3: Keyframe filtering validation
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

# Configure Chinese font support
def setup_chinese_font():
    """Setup Chinese font for matplotlib on Linux"""
    # Try common Chinese fonts on Linux
    chinese_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC', 
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'AR PL UMing CN',
        'AR PL UKai CN'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    chinese_font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font_found = font
            print(f"Using Chinese font: {font}")
            break
    
    if chinese_font_found:
        # Use Chinese font for Chinese characters, DejaVu Sans for English/numbers
        plt.rcParams['font.sans-serif'] = [chinese_font_found, 'DejaVu Sans', 'Arial']
    else:
        # Fallback: try to download and use WenQuanYi
        print("Warning: No Chinese font found, using system default")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

# Output directory
output_dir = Path(__file__).parent
fig1_path = output_dir / "fig_3_10_strategy_comparison.png"
fig2_path = output_dir / "fig_3_11_tau_sensitivity.png"

# ===== Figure 3.10: Strategy Comparison (FPS vs ATE) - SIMPLIFIED =====
print("Generating Figure 3.10: Strategy Comparison...")
df_strategies = pd.read_csv(output_dir / "eval_7scenes_long_3strategies_final/keyframe_strategy_7scenes_long.csv")

# Define strategy colors and markers
strategy_styles = {
    'A_Full': {'color': '#e74c3c', 'marker': 'o', 'size': 300, 'label': 'A_Full (基准)'},
    'B_Heuristic': {'color': '#3498db', 'marker': 's', 'size': 300, 'label': 'B_Heuristic (像素)'},
    'C_DINOReuse': {'color': '#2ecc71', 'marker': '^', 'size': 300}
}

fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=300)

# Get baseline FPS for speedup calculation
baseline_fps = df_strategies[df_strategies['strategy'] == 'A_Full']['fps'].mean()

# Plot by strategy groups
for strategy in ['A_Full', 'B_Heuristic', 'C_DINOReuse']:
    mask = df_strategies['strategy'] == strategy
    data = df_strategies[mask]
    
    if strategy == 'C_DINOReuse':
        # Group by tau and plot mean
        for tau_val in data['threshold_or_param'].unique():
            tau_data = data[data['threshold_or_param'] == tau_val]
            mean_fps = tau_data['fps'].mean()
            mean_ate = tau_data['ate'].mean()
            tau = tau_val.split('=')[1]
            label = f'C_DINOReuse (τ={tau})'
            
            ax1.scatter(mean_fps, mean_ate, 
                       color=strategy_styles[strategy]['color'], 
                       marker=strategy_styles[strategy]['marker'], 
                       s=strategy_styles[strategy]['size'], 
                       alpha=0.85, edgecolors='black', linewidth=2,
                       label=label, zorder=5)
            
            # Add speedup annotation for recommended tau
            if tau == '0.0005':
                speedup = mean_fps / baseline_fps
                ax1.annotate(f'{speedup:.1f}×', 
                            xy=(mean_fps, mean_ate),
                            xytext=(15, 15), textcoords='offset points',
                            fontsize=14, fontweight='bold', color='#2ecc71',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                     edgecolor='#2ecc71', linewidth=2),
                            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    else:
        # Plot mean for other strategies
        mean_fps = data['fps'].mean()
        mean_ate = data['ate'].mean()
        ax1.scatter(mean_fps, mean_ate, 
                   color=strategy_styles[strategy]['color'], 
                   marker=strategy_styles[strategy]['marker'], 
                   s=strategy_styles[strategy]['size'], 
                   alpha=0.85, edgecolors='black', linewidth=2,
                   label=strategy_styles[strategy]['label'], zorder=5)

ax1.set_xlabel('处理速度 (FPS)', fontsize=16, fontweight='bold')
ax1.set_ylabel('绝对轨迹误差 ATE (m)', fontsize=16, fontweight='bold')
ax1.set_title('图3.10 关键帧过滤策略对比', fontsize=18, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=13, framealpha=0.95, edgecolor='black', fancybox=False)
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.tick_params(labelsize=13)

# Set nice axis limits
ax1.set_xlim(1, 10)
ax1.set_ylim(0.85, 1.65)

plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig1_path}")
plt.close()

# ===== Figure 3.11: DINO Threshold Sensitivity - SIMPLIFIED =====
print("Generating Figure 3.11: DINO Threshold Sensitivity...")
df_sweep = pd.read_csv(output_dir / "dino_tau_sweep_chess_stairs_150_final/dino_threshold_sweep.csv")

# Filter only C_DINOReuse strategies
df_dino = df_sweep[df_sweep['strategy'] == 'C_DINOReuse'].copy()
df_dino['tau'] = df_dino['tau'].astype(float)

# Compute mean across scenes
mean_metrics = df_dino.groupby('tau').agg({
    'speedup_vs_full': 'mean',
    'ate_ratio_vs_full': 'mean'
}).reset_index()

# Create dual-axis plot
fig2, ax2_left = plt.subplots(figsize=(11, 7), dpi=300)
ax2_right = ax2_left.twinx()

# Left axis: Speedup (GREEN)
line1 = ax2_left.plot(mean_metrics['tau'], mean_metrics['speedup_vs_full'], 
                       color='#27ae60', marker='o', linewidth=3.5, markersize=12,
                       label='加速比', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax2_left.set_xlabel('阈值 τ', fontsize=16, fontweight='bold')
ax2_left.set_ylabel('加速比', fontsize=16, fontweight='bold', color='#27ae60')
ax2_left.tick_params(axis='y', labelcolor='#27ae60', labelsize=13)
ax2_left.tick_params(axis='x', labelsize=13)
ax2_left.grid(True, alpha=0.25, linestyle='--', zorder=0)

# Right axis: ATE ratio (RED)
line2 = ax2_right.plot(mean_metrics['tau'], mean_metrics['ate_ratio_vs_full'], 
                        color='#c0392b', marker='s', linewidth=3.5, markersize=12,
                        label='ATE比率', linestyle='--', 
                        markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax2_right.set_ylabel('ATE比率', fontsize=16, fontweight='bold', color='#c0392b')
ax2_right.tick_params(axis='y', labelcolor='#c0392b', labelsize=13)

# Highlight recommended tau=0.0005
recommended_tau = 0.0005
rec_data = mean_metrics[mean_metrics['tau'] == recommended_tau].iloc[0]
ax2_left.axvline(x=recommended_tau, color='#f39c12', linestyle='-', 
                alpha=0.6, linewidth=3, zorder=2)

# Add annotation box
ax2_left.annotate(f'推荐τ={recommended_tau}\n加速比={rec_data["speedup_vs_full"]:.2f}×', 
                 xy=(recommended_tau, rec_data['speedup_vs_full']),
                 xytext=(0.00065, 4.5), 
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff9e6', 
                          edgecolor='#f39c12', linewidth=3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', 
                                color='#f39c12', lw=2.5),
                 fontsize=14, fontweight='bold', ha='left')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, loc='upper left', fontsize=14, 
               framealpha=0.95, edgecolor='black', fancybox=False)

ax2_left.set_title('图3.11 阈值对性能的影响', fontsize=18, fontweight='bold', pad=20)

# Set y-axis limits
ax2_left.set_ylim(0, 13)
ax2_right.set_ylim(1.05, 1.28)

plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig2_path}")
plt.close()

print("\n=== Figure Generation Complete ===")
print(f"Figure 3.10: {fig1_path}")
print(f"Figure 3.11: {fig2_path}")
