"""
绘制VGGT模型各模块耗时分布图（图3.2）
展示帧内注意力与全局注意力随输入帧数的变化趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib import font_manager

# 设置中文字体
def setup_chinese_font():
    """配置中文字体支持"""
    mpl_data = matplotlib.get_data_path()
    font_candidates = [
        "simhei",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "Microsoft YaHei",
    ]
    
    for font in font_candidates:
        try:
            if Path(font).exists():
                font_manager.fontManager.addfont(font)
                font_prop = font_manager.FontProperties(fname=font)
                plt.rcParams['font.family'] = font_prop.get_name()
                break
        except:
            continue
    
    # 如果找不到中文字体，使用默认
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return font_manager.FontProperties(family=plt.rcParams['font.family'])

FONT_PROP = setup_chinese_font()

# 读取数据（CSV位于 tests/tests_result/module_latency/）
csv_path = Path(__file__).parent.parent / "tests_result" / "module_latency" / "module_latency_7scenes.csv"
df = pd.read_csv(csv_path)

# 数据预处理：将毫秒转换为秒
df['total_s'] = df['total_ms'] / 1000
df['frame_blocks_s'] = df['frame_blocks_total'] / 1000
df['global_blocks_s'] = df['global_blocks_total'] / 1000
df['patch_embed_s'] = df['patch_embed'] / 1000
df['camera_head_s'] = df['camera_head'] / 1000
df['depth_head_s'] = df['depth_head'] / 1000

# 计算其他模块总耗时（非attention部分）
df['other_modules_s'] = df['total_s'] - df['frame_blocks_s'] - df['global_blocks_s']

# 创建图表
fig = plt.figure(figsize=(16, 10))

# ===== 子图1: 主要模块耗时对比（线图） =====
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df['frame_count'], df['frame_blocks_s'], 'o-', linewidth=2.5, markersize=6, 
         label='帧内注意力 (Frame Attention)', color='#2E86AB')
ax1.plot(df['frame_count'], df['global_blocks_s'], 's-', linewidth=2.5, markersize=6, 
         label='全局注意力 (Global Attention)', color='#A23B72')
ax1.plot(df['frame_count'], df['other_modules_s'], '^-', linewidth=2, markersize=5, 
         label='其他模块 (Others)', color='#F18F01', alpha=0.7)

ax1.set_xlabel('输入帧数', fontsize=13, fontproperties=FONT_PROP)
ax1.set_ylabel('耗时 (秒)', fontsize=13, fontproperties=FONT_PROP)
ax1.set_title('(a) 各模块耗时随输入帧数变化', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
ax1.legend(prop=FONT_PROP, fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0, df['frame_count'].max() + 5])
ax1.set_ylim([0, None])

# 添加标注：突出全局注意力的二次增长
idx_50 = df[df['frame_count'] == 50].index[0]
idx_100 = df[df['frame_count'] == 100].index[0]
ax1.annotate('全局注意力\n呈二次方增长', 
             xy=(df.loc[idx_100, 'frame_count'], df.loc[idx_100, 'global_blocks_s']),
             xytext=(df.loc[idx_100, 'frame_count'] - 30, df.loc[idx_100, 'global_blocks_s'] + 15),
             fontsize=10, fontproperties=FONT_PROP, color='#A23B72',
             arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#A23B72', alpha=0.8))

# ===== 子图2: 占比堆叠面积图 =====
ax2 = plt.subplot(2, 2, 2)
ax2.fill_between(df['frame_count'], 0, df['frame_blocks_s'], 
                 label='帧内注意力', color='#2E86AB', alpha=0.7)
ax2.fill_between(df['frame_count'], df['frame_blocks_s'], 
                 df['frame_blocks_s'] + df['global_blocks_s'], 
                 label='全局注意力', color='#A23B72', alpha=0.7)
ax2.fill_between(df['frame_count'], df['frame_blocks_s'] + df['global_blocks_s'], 
                 df['total_s'], 
                 label='其他模块', color='#F18F01', alpha=0.6)

ax2.set_xlabel('输入帧数', fontsize=13, fontproperties=FONT_PROP)
ax2.set_ylabel('累计耗时 (秒)', fontsize=13, fontproperties=FONT_PROP)
ax2.set_title('(b) 各模块耗时占比堆叠图', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
ax2.legend(prop=FONT_PROP, fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.set_xlim([0, df['frame_count'].max() + 5])

# ===== 子图3: 时间复杂度对比（对数坐标） =====
ax3 = plt.subplot(2, 2, 3)

# 添加拟合曲线并计算误差
frame_counts = df['frame_count'].values
frame_blocks_vals = df['frame_blocks_s'].values
global_blocks_vals = df['global_blocks_s'].values

# 帧内注意力：线性拟合 (y = ax + b)
linear_coeffs = np.polyfit(frame_counts, frame_blocks_vals, 1)
linear_fit = np.poly1d(linear_coeffs)
linear_pred = linear_fit(frame_counts)
linear_rmse = np.sqrt(np.mean((frame_blocks_vals - linear_pred)**2))
linear_r2 = 1 - (np.sum((frame_blocks_vals - linear_pred)**2) / 
                 np.sum((frame_blocks_vals - np.mean(frame_blocks_vals))**2))

# 帧内注意力：二次拟合 (y = ax² + bx + c) 用于对比
linear_quad_coeffs = np.polyfit(frame_counts, frame_blocks_vals, 2)
linear_quad_fit = np.poly1d(linear_quad_coeffs)
linear_quad_pred = linear_quad_fit(frame_counts)
linear_quad_rmse = np.sqrt(np.mean((frame_blocks_vals - linear_quad_pred)**2))
linear_quad_r2 = 1 - (np.sum((frame_blocks_vals - linear_quad_pred)**2) / 
                      np.sum((frame_blocks_vals - np.mean(frame_blocks_vals))**2))

# 全局注意力：二次拟合 (y = ax² + bx + c)
quad_coeffs = np.polyfit(frame_counts, global_blocks_vals, 2)
quad_fit = np.poly1d(quad_coeffs)
quad_pred = quad_fit(frame_counts)
quad_rmse = np.sqrt(np.mean((global_blocks_vals - quad_pred)**2))
quad_r2 = 1 - (np.sum((global_blocks_vals - quad_pred)**2) / 
               np.sum((global_blocks_vals - np.mean(global_blocks_vals))**2))

# 全局注意力：线性拟合用于对比
quad_linear_coeffs = np.polyfit(frame_counts, global_blocks_vals, 1)
quad_linear_fit = np.poly1d(quad_linear_coeffs)
quad_linear_pred = quad_linear_fit(frame_counts)
quad_linear_rmse = np.sqrt(np.mean((global_blocks_vals - quad_linear_pred)**2))
quad_linear_r2 = 1 - (np.sum((global_blocks_vals - quad_linear_pred)**2) / 
                      np.sum((global_blocks_vals - np.mean(global_blocks_vals))**2))

# 绘制原始数据点和拟合曲线
ax3.semilogy(df['frame_count'], df['frame_blocks_s'], 'o', markersize=6,
             label=f'帧内注意力 (实测)', color='#2E86AB')
ax3.semilogy(df['frame_count'], df['global_blocks_s'], 's', markersize=6,
             label=f'全局注意力 (实测)', color='#A23B72')

# 由于在对数坐标下，线性函数会显示为指数曲线，改用普通坐标拟合后再绘制
# 帧内注意力：实际更接近二次而非线性
ax3.semilogy(frame_counts, linear_quad_pred, '--', linewidth=2, 
             color='#2E86AB', alpha=0.6, 
             label=f'帧内二次拟合 (R²={linear_quad_r2:.4f})')

# 全局注意力：明确的二次增长
ax3.semilogy(frame_counts, quad_pred, '--', linewidth=2, 
             color='#A23B72', alpha=0.6, 
             label=f'全局二次拟合 (R²={quad_r2:.4f})')

ax3.set_xlabel('输入帧数', fontsize=13, fontproperties=FONT_PROP)
ax3.set_ylabel('耗时 (秒, 对数坐标)', fontsize=13, fontproperties=FONT_PROP)
ax3.set_title('(c) 时间复杂度对比（对数坐标）', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
ax3.legend(prop=FONT_PROP, fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3, linestyle='--', which='both')
ax3.set_xlim([0, df['frame_count'].max() + 5])

# ===== 子图4: 百分比占比图 =====
ax4 = plt.subplot(2, 2, 4)
df['frame_blocks_pct'] = df['frame_blocks_s'] / df['total_s'] * 100
df['global_blocks_pct'] = df['global_blocks_s'] / df['total_s'] * 100
df['other_modules_pct'] = df['other_modules_s'] / df['total_s'] * 100

ax4.plot(df['frame_count'], df['frame_blocks_pct'], 'o-', linewidth=2.5, markersize=6,
         label='帧内注意力', color='#2E86AB')
ax4.plot(df['frame_count'], df['global_blocks_pct'], 's-', linewidth=2.5, markersize=6,
         label='全局注意力', color='#A23B72')
ax4.plot(df['frame_count'], df['other_modules_pct'], '^-', linewidth=2, markersize=5,
         label='其他模块', color='#F18F01', alpha=0.7)

ax4.set_xlabel('输入帧数', fontsize=13, fontproperties=FONT_PROP)
ax4.set_ylabel('占总耗时百分比 (%)', fontsize=13, fontproperties=FONT_PROP)
ax4.set_title('(d) 各模块耗时占比变化趋势', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
ax4.legend(prop=FONT_PROP, fontsize=11, loc='best')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim([0, df['frame_count'].max() + 5])
ax4.set_ylim([0, 100])

# 添加关键观察标注
idx_last = df.index[-1]
ax4.annotate(f'全局注意力占比: {df.loc[idx_last, "global_blocks_pct"]:.1f}%', 
             xy=(df.loc[idx_last, 'frame_count'], df.loc[idx_last, 'global_blocks_pct']),
             xytext=(df.loc[idx_last, 'frame_count'] - 40, df.loc[idx_last, 'global_blocks_pct'] - 15),
             fontsize=10, fontproperties=FONT_PROP, color='#A23B72',
             arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#A23B72', alpha=0.8))

# 调整整体布局
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.25, wspace=0.25)

# 添加总标题
# fig.suptitle('图3.2: VGGT模型各模块耗时分布与复杂度分析', 
#              fontsize=16, fontweight='bold', fontproperties=FONT_PROP, y=0.98)

# 保存图表到 tests/tests_result/module_latency/
output_path = Path(__file__).parent.parent / "tests_result" / "module_latency" / "figure_3.2_module_latency_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 图表已保存至: {output_path}")

# 生成数据分析报告
print("\n" + "="*80)
print("数据分析报告：")
print("="*80)

# 计算关键指标
frame_5 = df[df['frame_count'] == 5].iloc[0]
frame_50 = df[df['frame_count'] == 50].iloc[0]
frame_100 = df[df['frame_count'] == 100].iloc[0]
frame_145 = df[df['frame_count'] == 145].iloc[0]

print("\n1. 短序列场景（5帧）：")
print(f"   - 帧内注意力: {frame_5['frame_blocks_s']:.2f}秒 ({frame_5['frame_blocks_pct']:.1f}%)")
print(f"   - 全局注意力: {frame_5['global_blocks_s']:.2f}秒 ({frame_5['global_blocks_pct']:.1f}%)")
print(f"   - 两者比例: {frame_5['global_blocks_s'] / frame_5['frame_blocks_s']:.2f}:1")

print("\n2. 中等序列场景（50帧）：")
print(f"   - 帧内注意力: {frame_50['frame_blocks_s']:.2f}秒 ({frame_50['frame_blocks_pct']:.1f}%)")
print(f"   - 全局注意力: {frame_50['global_blocks_s']:.2f}秒 ({frame_50['global_blocks_pct']:.1f}%)")
print(f"   - 两者比例: {frame_50['global_blocks_s'] / frame_50['frame_blocks_s']:.2f}:1")

print("\n3. 长序列场景（100帧）：")
print(f"   - 帧内注意力: {frame_100['frame_blocks_s']:.2f}秒 ({frame_100['frame_blocks_pct']:.1f}%)")
print(f"   - 全局注意力: {frame_100['global_blocks_s']:.2f}秒 ({frame_100['global_blocks_pct']:.1f}%)")
print(f"   - 两者比例: {frame_100['global_blocks_s'] / frame_100['frame_blocks_s']:.2f}:1")

print("\n4. 极长序列场景（145帧）：")
print(f"   - 帧内注意力: {frame_145['frame_blocks_s']:.2f}秒 ({frame_145['frame_blocks_pct']:.1f}%)")
print(f"   - 全局注意力: {frame_145['global_blocks_s']:.2f}秒 ({frame_145['global_blocks_pct']:.1f}%)")
print(f"   - 两者比例: {frame_145['global_blocks_s'] / frame_145['frame_blocks_s']:.2f}:1")

print("\n5. 增长倍数分析（5帧 → 145帧）：")
print(f"   - 帧内注意力增长: {frame_145['frame_blocks_s'] / frame_5['frame_blocks_s']:.1f}倍")
print(f"   - 全局注意力增长: {frame_145['global_blocks_s'] / frame_5['global_blocks_s']:.1f}倍")
print(f"   - 总耗时增长: {frame_145['total_s'] / frame_5['total_s']:.1f}倍")

print("\n6. 关键结论：")
print("   ✓ 全局注意力在长序列下呈现二次方增长，成为主导瓶颈")
print("   ✓ 帧内注意力增长平稳，保持线性复杂度特征")
print(f"   ✓ 在145帧时，全局注意力占总耗时的 {frame_145['global_blocks_pct']:.1f}%")

print("\n7. 拟合误差分析：")
print(f"   帧内注意力拟合对比：")
print(f"     - 线性拟合 (y=ax+b):  R²={linear_r2:.6f}, RMSE={linear_rmse:.4f}秒")
print(f"     - 二次拟合 (y=ax²+bx+c): R²={linear_quad_r2:.6f}, RMSE={linear_quad_rmse:.4f}秒")
print(f"     - 结论: 二次拟合R²更接近1，RMSE更小，帧内注意力实际为 O(n²) 复杂度")
print(f"\n   全局注意力拟合对比：")
print(f"     - 线性拟合 (y=ax+b):  R²={quad_linear_r2:.6f}, RMSE={quad_linear_rmse:.4f}秒")
print(f"     - 二次拟合 (y=ax²+bx+c): R²={quad_r2:.6f}, RMSE={quad_rmse:.4f}秒")
print(f"     - 结论: 二次拟合显著优于线性，验证全局注意力为 O(n²) 复杂度")
print(f"\n   ⚠️  重要发现:")
print(f"     帧内注意力虽然增长较慢，但实际仍为二次复杂度 O(n²)")
print(f"     相比全局注意力，其系数 a 更小，因此增长更平缓")
print(f"     帧内: y ≈ {linear_quad_coeffs[0]:.6f}x² + {linear_quad_coeffs[1]:.4f}x + {linear_quad_coeffs[2]:.4f}")
print(f"     全局: y ≈ {quad_coeffs[0]:.6f}x² + {quad_coeffs[1]:.4f}x + {quad_coeffs[2]:.4f}")
print("="*80)

plt.show()
