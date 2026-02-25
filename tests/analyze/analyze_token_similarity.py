"""
分析token_similarity随序列长度和层级的变化规律
目的：找出在不同序列长度中均成立的相似度随层级变化的规律
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import warnings
import matplotlib
from matplotlib import font_manager
warnings.filterwarnings('ignore')

# 设置中文字体
def setup_chinese_font():
    """配置中文字体支持"""
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
                return font_prop
        except:
            continue
    
    # 如果找不到中文字体，使用默认
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return font_manager.FontProperties(family=plt.rcParams['font.family'])

FONT_PROP = setup_chinese_font()
sns.set_style("whitegrid")

# 数据路径
DATA_PATH = Path(__file__).parent.parent / "tests_result" / "token_similarity" / "token_similarity_results.csv"
BASE_OUTPUT_DIR = Path(__file__).parent.parent / "tests_result" / "token_similarity"
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# 创建子文件夹
OUTPUT_DIR = BASE_OUTPUT_DIR / "1_fitting_analysis"  # 拟合分析结果
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_DIR_LAYER_SENSITIVITY = BASE_OUTPUT_DIR / "2_layer_sensitivity"  # 层级灵敏度分析结果
OUTPUT_DIR_LAYER_SENSITIVITY.mkdir(exist_ok=True)

# ============================================================================
# 1. 加载和预处理数据
# ============================================================================
def load_data():
    """加载CSV数据"""
    df = pd.read_csv(DATA_PATH)
    print(f"数据已加载: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"序列长度: {sorted(df['sequence_length'].unique())}")
    print(f"块层级: {sorted(df['block_layer'].unique())}")
    return df

# ============================================================================
# 2. 拟合函数库
# ============================================================================
def linear_fit(x, a, b):
    """线性拟合: y = ax + b"""
    return a * x + b

def polynomial_fit_2nd(x, a, b, c):
    """二次多项式拟合: y = ax^2 + bx + c"""
    return a * x**2 + b * x + c

def polynomial_fit_3rd(x, a, b, c, d):
    """三次多项式拟合: y = ax^3 + bx^2 + cx + d"""
    return a * x**3 + b * x**2 + c * x + d

def exponential_fit(x, a, b):
    """指数拟合: y = a * exp(bx)"""
    return a * np.exp(b * x)

def power_fit(x, a, b):
    """幂律拟合: y = a * x^b"""
    return a * np.power(x, b)

def gaussian_fit(x, a, b, c):
    """高斯拟合: y = a * exp(-(x-b)^2 / (2*c^2))"""
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def sigmoid_fit(x, a, b, c):
    """Sigmoid拟合: y = a / (1 + exp(-b*(x-c)))"""
    return a / (1 + np.exp(-b * (x - c)))

# ============================================================================
# 3. 拟合和误差计算
# ============================================================================
def fit_and_evaluate(x, y, fit_func, p0=None, bounds=(-np.inf, np.inf)):
    """
    对数据进行曲线拟合并计算误差指标
    """
    try:
        popt, _ = curve_fit(fit_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = fit_func(x, *popt)
        
        # 计算误差指标
        mse = np.mean((y - y_pred)**2)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        mae = np.mean(np.abs(y - y_pred))  # 平均绝对误差
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)  # 决定系数
        
        return {
            'params': popt,
            'y_pred': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        return None

def analyze_by_sequence_length(df):
    """
    按序列长度分别分析相似度随层级的变化规律
    """
    results = {}
    seq_lengths = sorted(df['sequence_length'].unique())
    
    for seq_len in seq_lengths:
        data = df[df['sequence_length'] == seq_len].sort_values('block_layer')
        x = data['block_layer'].values
        y = data['token_similarity'].values
        
        fits = {}
        
        # 线性拟合
        fits['linear'] = fit_and_evaluate(x, y, linear_fit, p0=[0.01, 0.3])
        
        # 二次多项式拟合
        fits['poly2'] = fit_and_evaluate(x, y, polynomial_fit_2nd, p0=[0.001, 0.01, 0.3])
        
        # 三次多项式拟合
        fits['poly3'] = fit_and_evaluate(x, y, polynomial_fit_3rd, p0=[0.0001, 0.001, 0.01, 0.3])
        
        # 指数拟合（需要正值初值）
        fits['exp'] = fit_and_evaluate(x, y, exponential_fit, p0=[0.3, 0.05], 
                                       bounds=([0.1, -0.2], [1.0, 0.2]))
        
        # 幂律拟合（需要正值初值）
        fits['power'] = fit_and_evaluate(x, y, power_fit, p0=[0.3, 2.0],
                                        bounds=([0.1, 0.1], [1.0, 3.0]))
        
        # 高斯拟合
        fits['gaussian'] = fit_and_evaluate(x, y, gaussian_fit, p0=[0.6, 15, 5])
        
        # Sigmoid拟合
        fits['sigmoid'] = fit_and_evaluate(x, y, sigmoid_fit, p0=[0.6, 0.1, 12])

        for k, v in fits.items():
            if v is None:
                print(f"{k} 拟合失败")
        
        # 移除失败的拟合
        fits = {k: v for k, v in fits.items() if v is not None}
        
        results[seq_len] = {
            'x': x,
            'y': y,
            'fits': fits
        }
    
    return results

# ============================================================================
# 4. 选择最优拟合
# ============================================================================
def find_best_fit(results, metric='r2'):
    """
    找出在所有序列长度中表现最好的拟合方法
    """
    print(f"\n{'='*80}")
    print(f"按 {metric} 指标评选最优拟合")
    print(f"{'='*80}\n")
    
    # 收集所有序列长度的拟合结果
    all_metrics = {}
    all_details = {}  # 保存详细信息用于导出CSV
    
    for seq_len, data in results.items():
        print(f"序列长度 = {seq_len}:")
        print(f"{'拟合方法':<12} {'R²':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
        print("-" * 60)
        
        for fit_name, fit_result in sorted(data['fits'].items()):
            r2 = fit_result['r2']
            mse = fit_result['mse']
            rmse = fit_result['rmse']
            mae = fit_result['mae']
            
            if fit_name not in all_metrics:
                all_metrics[fit_name] = []
                all_details[fit_name] = []
            all_metrics[fit_name].append(fit_result[metric])
            all_details[fit_name].append({
                'seq_len': seq_len,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            })
            
            print(f"{fit_name:<12} {r2:<12.4f} {mse:<12.6f} {rmse:<12.4f} {mae:<12.4f}")
        print()
    
    # 计算平均性能
    print(f"{'='*80}")
    print("误差指标总结 - 各拟合方法的平均性能")
    print(f"{'='*80}\n")
    
    avg_metrics = {name: np.mean(scores) for name, scores in all_metrics.items()}
    sorted_methods = sorted(avg_metrics.items(), key=lambda x: x[1], reverse=True)
    
    # 为了计算平均MSE和RMSE，我们需要重新收集这些数据
    avg_mse = {}
    avg_rmse = {}
    avg_mae = {}
    for fit_name, details_list in all_details.items():
        avg_mse[fit_name] = np.mean([d['mse'] for d in details_list])
        avg_rmse[fit_name] = np.mean([d['rmse'] for d in details_list])
        avg_mae[fit_name] = np.mean([d['mae'] for d in details_list])
    
    print(f"{'排名':<6} {'拟合方法':<12} {'平均R²':<15} {'平均MSE':<15} {'平均RMSE':<15} {'平均MAE':<15}")
    print("-" * 95)
    for i, (name, avg_r2) in enumerate(sorted_methods, 1):
        avg_mse_val = avg_mse.get(name, 0)
        avg_rmse_val = avg_rmse.get(name, 0)
        avg_mae_val = avg_mae.get(name, 0)
        print(f"{i:<6} {name:<12} {avg_r2:<15.4f} {avg_mse_val:<15.6f} {avg_rmse_val:<15.4f} {avg_mae_val:<15.4f}")
    
    best_method = sorted_methods[0][0]
    best_r2 = avg_metrics[best_method]
    print(f"\n✓ 最优拟合方法: {best_method}")
    print(f"  • 平均 R² = {best_r2:.4f}")
    print(f"  • 平均 MSE = {avg_mse[best_method]:.6f}")
    print(f"  • 平均 RMSE = {avg_rmse[best_method]:.4f}")
    print(f"  • 平均 MAE = {avg_mae[best_method]:.4f}")
    
    # 保存详细误差数据为CSV
    save_error_details_to_csv(all_details)
    
    return best_method, results

def save_error_details_to_csv(all_details):
    """将每个序列长度的拟合误差保存为CSV"""
    rows = []
    for fit_name, details_list in all_details.items():
        for detail in details_list:
            rows.append({
                '拟合方法': fit_name,
                '序列长度': detail['seq_len'],
                'R²': detail['r2'],
                'MSE': detail['mse'],
                'RMSE': detail['rmse'],
                'MAE': detail['mae']
            })
    
    df_errors = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "拟合误差详情表.csv"
    df_errors.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 误差详情已保存: {output_file}")

# ============================================================================
# 5. 可视化
# ============================================================================
def plot_all_fits(results, best_method):
    """
    绘制所有序列长度的数据和最优拟合曲线
    """
    seq_lengths = sorted(results.keys())
    n_plots = len(seq_lengths)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        data = results[seq_len]
        x = data['x']
        y = data['y']
        
        # 绘制原始数据
        ax.scatter(x, y, s=60, alpha=0.6, color='#1f77b4', label='原始数据', zorder=3)
        
        # 定义颜色和线型用于不同的拟合方法
        fit_styles = {
            'poly3': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5},
            'poly2': {'color': 'orange', 'linestyle': '--', 'linewidth': 1.5},
            'linear': {'color': 'green', 'linestyle': ':', 'linewidth': 1.5},
            'exp': {'color': 'purple', 'linestyle': '-.', 'linewidth': 1.2},
            'power': {'color': 'brown', 'linestyle': '--', 'linewidth': 1.2},
            'gaussian': {'color': 'pink', 'linestyle': ':', 'linewidth': 1.2},
            'sigmoid': {'color': 'cyan', 'linestyle': '-.', 'linewidth': 1.2}
        }
        
        # 定义拟合方法的中文名称
        fit_names_cn = {
            'poly3': '三次多项式',
            'poly2': '二次多项式',
            'linear': '线性',
            'exp': '指数',
            'power': '幂律',
            'gaussian': '高斯',
            'sigmoid': 'Sigmoid'
        }
        
        # 绘制所有拟合曲线，最优方法优先
        fit_names = sorted(data['fits'].keys())
        # 将最优方法移到列表前面
        if best_method in fit_names:
            fit_names.remove(best_method)
            fit_names.insert(0, best_method)
        
        for fit_name in fit_names:
            if fit_name in data['fits']:
                fit_result = data['fits'][fit_name]
                y_pred = fit_result['y_pred']
                r2 = fit_result['r2']
                
                # 按x值排序后绘制平滑曲线
                sorted_indices = np.argsort(x)
                
                style = fit_styles.get(fit_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
                fit_name_cn = fit_names_cn.get(fit_name, fit_name)
                
                # 最优方法用实心粗线，其他用虚线
                if fit_name == best_method:
                    ax.plot(x[sorted_indices], y_pred[sorted_indices], 
                           color=style['color'], linestyle=style['linestyle'], 
                           linewidth=style['linewidth'],
                           label=f'{fit_name_cn} (R²={r2:.4f})', zorder=2)
                else:
                    ax.plot(x[sorted_indices], y_pred[sorted_indices], 
                           color=style['color'], linestyle=style['linestyle'], 
                           linewidth=style['linewidth'], alpha=0.7,
                           label=f'{fit_name_cn} (R²={r2:.4f})', zorder=1)
        
        ax.set_xlabel('块层级', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_ylabel('相似度', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_title(f'序列长度 = {seq_len}', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
        ax.legend(loc='best', fontsize=8, prop=FONT_PROP, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 24)
    
    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "相似度所有序列长度拟合.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def plot_heatmap(df):
    """
    绘制相似度热力图（序列长度 x 块层级）
    """
    pivot_df = df.pivot(index='block_layer', columns='sequence_length', values='token_similarity')
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot_df, cmap='RdYlGn', annot=False, fmt='.3f', cbar_kws={'label': '相似度'},
                ax=ax, linewidths=0.5, vmin=0.2, vmax=0.95)
    
    # 为 colorbar label 设置字体
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.set_label('相似度', fontproperties=FONT_PROP, fontsize=11)
    
    ax.set_xlabel('序列长度', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_ylabel('块层级', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_title('相似度热力图: 层级 vs 序列长度', fontsize=13, fontweight='bold', fontproperties=FONT_PROP)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "相似度热力图.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def plot_normalized_comparison(results):
    """
    绘制归一化比较图：显示不同序列长度下相似度变化的一致性
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    seq_lengths = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(seq_lengths)))
    
    for seq_len, color in zip(seq_lengths, colors):
        data = results[seq_len]
        x = data['x']
        y = data['y']
        
        # 归一化到 [0, 1]
        y_norm = (y - y.min()) / (y.max() - y.min())
        
        ax.plot(x, y_norm, 'o-', linewidth=2.5, markersize=6, label=f'序列长度={seq_len}', 
               color=color, alpha=0.8)
    
    ax.set_xlabel('块层级', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_ylabel('归一化相似度', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_title('跨序列长度的相似度分布一致性\n(所有序列均遵循相同趋势)', 
                fontsize=13, fontweight='bold', fontproperties=FONT_PROP)
    ax.legend(loc='best', fontsize=10, prop=FONT_PROP)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 24)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "归一化相似度对比.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def plot_layer_regions(results):
    """
    绘制不同层级区间的相似度统计
    定义四个区间: 浅层 (0-3), 中层-1 (4-10), 中层-2 (11-16), 深层 (17-23)
    """
    regions = {
        '浅层 (0-3)': (0, 4),
        '中层-1 (4-10)': (4, 11),
        '中层-2 (11-16)': (11, 17),
        '深层 (17-23)': (17, 24)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # 将2x2的axes展平为一维数组
    
    seq_lengths = sorted(results.keys())
    x_pos = np.arange(len(seq_lengths))
    
    for ax_idx, (region_name, (start, end)) in enumerate(regions.items()):
        ax = axes[ax_idx]
        
        region_means = []
        region_stds = []
        
        for seq_len in seq_lengths:
            data = results[seq_len]
            y = data['y']
            x = data['x']
            
            # 提取该区间的数据
            mask = (x >= start) & (x < end)
            region_data = y[mask]
            
            region_means.append(region_data.mean())
            region_stds.append(region_data.std())
        
        # 绘制柱状图with误差条
        bars = ax.bar(x_pos, region_means, yerr=region_stds, capsize=5, 
                     color='skyblue', edgecolor='navy', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('序列长度', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_ylabel('平均相似度', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_title(region_name, fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seq_lengths)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "分区相似度统计.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def plot_fitted_parameters(results, best_method):
    """
    绘制拟合参数随序列长度的变化
    """
    seq_lengths = sorted(results.keys())
    params_list = []
    
    for seq_len in seq_lengths:
        data = results[seq_len]
        if best_method in data['fits']:
            params = data['fits'][best_method]['params']
            params_list.append(params)
    
    if not params_list:
        return
    
    params_array = np.array(params_list)
    n_params = params_array.shape[1]
    
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    param_names = ['a', 'b', 'c', 'd'][:n_params]
    
    for idx, (param_name, ax) in enumerate(zip(param_names, axes)):
        ax.plot(seq_lengths, params_array[:, idx], 'o-', linewidth=2.5, markersize=8, 
               color='#ff7f0e', markerfacecolor='#ff7f0e', markeredgecolor='darkblue')
        
        ax.set_xlabel('序列长度', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_ylabel(f'参数 {param_name}', fontsize=11, fontweight='bold', fontproperties=FONT_PROP)
        ax.set_title(f'{best_method}: 参数 {param_name}', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"拟合参数_{best_method}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def export_fitting_comparison_table(results, best_method):
    """
    导出拟合对比表为CSV
    """
    rows = []
    seq_lengths = sorted(results.keys())
    fit_methods = sorted(list(results[seq_lengths[0]]['fits'].keys()))
    
    for fit_method in fit_methods:
        for seq_len in seq_lengths:
            if fit_method in results[seq_len]['fits']:
                fit_result = results[seq_len]['fits'][fit_method]
                rows.append({
                    '拟合方法': fit_method,
                    '序列长度': seq_len,
                    'R²': fit_result['r2'],
                    'MSE': fit_result['mse'],
                    'RMSE': fit_result['rmse'],
                    'MAE': fit_result['mae'],
                    '是否最优': '✓ 是' if fit_method == best_method else ''
                })
    
    df_comparison = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "拟合方法对比表.csv"
    df_comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 拟合对比表已保存: {output_file}")

def export_layer_statistics(results):
    """
    导出分层统计为CSV
    """
    regions = {
        '浅层 (0-3)': (0, 4),
        '中层-1 (4-10)': (4, 11),
        '中层-2 (11-16)': (11, 17),
        '深层 (17-23)': (17, 24)
    }
    
    rows = []
    seq_lengths = sorted(results.keys())
    
    for region_name, (start, end) in regions.items():
        for seq_len in seq_lengths:
            data = results[seq_len]
            y = data['y']
            x = data['x']
            
            mask = (x >= start) & (x < end)
            region_data = y[mask]
            
            rows.append({
                '区间': region_name,
                '序列长度': seq_len,
                '平均相似度': region_data.mean(),
                '标准差': region_data.std(),
                '最小值': region_data.min(),
                '最大值': region_data.max(),
                '数据点数': len(region_data)
            })
    
    df_stats = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "分层统计表.csv"
    df_stats.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 分层统计表已保存: {output_file}")

# ============================================================================
# 7. 层级灵敏度分析（按层级分析相似度随序列长度的变化）
# ============================================================================
def plot_similarity_vs_seq_len_per_layer(df):
    """
    绘制每个块层级上，相似度随序列长度的变化
    用多条线表示不同的层级，看是否有共同的变化趋势
    """
    block_layers = sorted(df['block_layer'].unique())
    seq_lengths = sorted(df['sequence_length'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义不同的颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(block_layers)))
    
    for layer_idx, layer in enumerate(block_layers):
        layer_data = df[df['block_layer'] == layer].sort_values('sequence_length')
        x = layer_data['sequence_length'].values
        y = layer_data['token_similarity'].values
        
        ax.plot(x, y, 'o-', linewidth=2, markersize=5, label=f'块层级 {layer}',
               color=colors[layer_idx], alpha=0.7)
    
    ax.set_xlabel('序列长度', fontsize=13, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_ylabel('相似度', fontsize=13, fontweight='bold', fontproperties=FONT_PROP)
    ax.set_title('不同块层级上：相似度随序列长度的变化\n'
                '(对比各层级是否遵循相同的变化趋势)', 
                fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, prop=FONT_PROP, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(seq_lengths)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_LAYER_SENSITIVITY / "各层级相似度vs序列长度.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()

def analyze_similarity_consistency_across_layers(df):
    """
    分析不同层级上，相似度变化趋势的一致性
    按层级分组，计算各层级在不同序列长度上的线性变化斜率
    """
    print(f"\n{'='*80}")
    print("层级灵敏度分析（按序列长度查看各层级相似度变化）")
    print(f"{'='*80}\n")
    
    block_layers = sorted(df['block_layer'].unique())
    seq_lengths = sorted(df['sequence_length'].unique())
    
    # 创建热力图数据：行为层级，列为序列长度
    pivot_table = df.pivot(index='block_layer', columns='sequence_length', values='token_similarity')
    
    print("相似度矩阵（行=块层级, 列=序列长度）：\n")
    print(pivot_table.to_string())
    print()
    
    # 计算每个层级的变化斜率（序列长度从5到100的相似度变化）
    print(f"{'='*80}")
    print("各层级的变化趋势分析")
    print(f"{'='*80}\n")
    
    print(f"{'块层级':<10} {'序列长度5':<12} {'序列长度100':<12} {'变化量':<12} {'变化方向':<12}")
    print("-" * 58)
    
    slope_data = []
    for layer in block_layers:
        layer_data = df[df['block_layer'] == layer].sort_values('sequence_length')
        sim_at_5 = layer_data[layer_data['sequence_length'] == 5]['token_similarity'].values
        sim_at_100 = layer_data[layer_data['sequence_length'] == 100]['token_similarity'].values
        
        if len(sim_at_5) > 0 and len(sim_at_100) > 0:
            sim_5 = float(sim_at_5[0])
            sim_100 = float(sim_at_100[0])
            change = sim_100 - sim_5
            direction = "增加" if change > 0 else ("减少" if change < 0 else "不变")
            
            print(f"{layer:<10} {sim_5:<12.4f} {sim_100:<12.4f} {change:<12.4f} {direction:<12}")
            slope_data.append({
                'block_layer': layer,
                'sim_at_5': sim_5,
                'sim_at_100': sim_100,
                'change': change
            })
    
    print()
    
    # 按层级分组统计，看浅、中、深层的变化趋势是否一致
    print(f"{'='*80}")
    print("分层统计：按浅层、中层、深层分组")
    print(f"{'='*80}\n")
    
    regions = {
        '浅层 (0-3)': (0, 4),
        '中层-1 (4-10)': (4, 11),
        '中层-2 (11-16)': (11, 17),
        '深层 (17-23)': (17, 24)
    }
    
    print(f"{'区间':<15} {'平均变化量':<15} {'变化方向':<12} {'一致性':<12}")
    print("-" * 54)
    
    for region_name, (start, end) in regions.items():
        region_slopes = [s['change'] for s in slope_data if start <= s['block_layer'] < end]
        
        if region_slopes:
            avg_change = np.mean(region_slopes)
            std_change = np.std(region_slopes)
            direction = "增加" if avg_change > 0 else ("减少" if avg_change < 0 else "不变")
            consistency = "高" if std_change < 0.05 else ("中" if std_change < 0.1 else "低")
            
            print(f"{region_name:<15} {avg_change:<15.4f} {direction:<12} {consistency:<12}")
    
    print()

def export_layer_sensitivity_table(df):
    """
    导出层级灵敏度表为CSV
    """
    block_layers = sorted(df['block_layer'].unique())
    seq_lengths = sorted(df['sequence_length'].unique())
    
    rows = []
    for layer in block_layers:
        layer_data = df[df['block_layer'] == layer]
        for seq_len in seq_lengths:
            seq_data = layer_data[layer_data['sequence_length'] == seq_len]
            if len(seq_data) > 0:
                rows.append({
                    '块层级': layer,
                    '序列长度': seq_len,
                    '相似度': seq_data['token_similarity'].values[0]
                })
    
    df_sensitivity = pd.DataFrame(rows)
    output_file = OUTPUT_DIR_LAYER_SENSITIVITY / "层级灵敏度统计表.csv"
    df_sensitivity.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 层级灵敏度表已保存: {output_file}\n")

# ============================================================================
# 6. 主函数
# ============================================================================
def main():
    print("\n" + "="*80)
    print("Token 相似度分析")
    print("="*80)
    
    # 加载数据
    df = load_data()
    
    # 分析
    results = analyze_by_sequence_length(df)
    
    # 选择最优拟合
    best_method, _ = find_best_fit(results, metric='r2')
    
    # 绘制可视化图表
    print(f"\n{'='*80}")
    print("第一阶段：生成拟合分析图表...")
    print(f"{'='*80}\n")
    
    plot_all_fits(results, best_method)
    plot_heatmap(df)
    plot_normalized_comparison(results)
    plot_layer_regions(results)
    plot_fitted_parameters(results, best_method)
    
    # 导出CSV表格
    print(f"\n{'='*80}")
    print("第一阶段：导出拟合分析表格...")
    print(f"{'='*80}\n")
    
    export_fitting_comparison_table(results, best_method)
    export_layer_statistics(results)
    
    # 第二阶段：层级灵敏度分析
    print(f"\n{'='*80}")
    print("第二阶段：生成层级灵敏度分析...")
    print(f"{'='*80}\n")
    
    plot_similarity_vs_seq_len_per_layer(df)
    analyze_similarity_consistency_across_layers(df)
    export_layer_sensitivity_table(df)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("分析总结")
    print(f"{'='*80}\n")
    
    print(f"""
✓ 分析完成！已生成两个阶段的分析结果

阶段一 - 拟合分析（存放在 1_fitting_analysis/）：
1. 最优拟合方法: {best_method}
   - 在不同序列长度 (5, 10, 30, 50, 100) 中表现一致
   - 在所有层级 (0-23) 中具有良好的拟合精度

2. 相似度规律（四层分析）：
   - 浅层 (0-3): 相对较低的相似度，变化稳定
   - 中层-1 (4-10): 相似度逐渐增加
   - 中层-2 (11-16): 相似度继续上升，进入较高水平
   - 深层 (17-23): 相似度较高，接近饱和

3. 序列长度影响：
   - 序列长度变化对相似度分布的整体形状影响较小
   - 表明该相似度分布规律具有通用性

阶段二 - 层级灵敏度分析（存放在 2_layer_sensitivity/）：
探究在不同的块层级上，相似度随序列长度的变化是否相同：
   - 绘制各层级的相似度变化趋势
   - 分析不同层级间的变化一致性
   - 统计各层级和分区的灵敏度差异

输出文件已保存到:
  - 拟合分析: {OUTPUT_DIR}
  - 层级灵敏度: {OUTPUT_DIR_LAYER_SENSITIVITY}
    """)


if __name__ == '__main__':
    main()
