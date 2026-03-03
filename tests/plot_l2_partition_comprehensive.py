"""
综合可视化分析：基于L2范数的Token划分有效性验证
=================================================================
多角度展示：
1. CD(Chamfer Distance)对比分析
2. 所有指标(CD/ATE/ARE)综合对比
3. 性能-成本权衡(Pareto)分析
4. 改进百分比热力图
5. 详细统计表格
6. 时间消耗分析

特点：
- 完整的中文字体支持
- 出版级别的图表质量(300+DPI)
- 自动检测并处理空缺数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def setup_chinese_font():
    """设置中文字体，确保无乱码"""
    try:
        # 尝试使用系统字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        matplotlib.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Courier New']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 10
    except:
        pass


def load_results(csv_path):
    """加载CSV结果文件"""
    df = pd.read_csv(csv_path)
    print(f"✓ 成功加载 {len(df)} 条测试记录")
    print(f"  帧数: {sorted(df['frame_count'].unique())}")
    print(f"  方法: {df['method'].unique()}")
    return df


def create_cd_comparison_chart(df, output_dir):
    """图1: CD指标对比（主要指标）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('L2范数Token划分 vs 固定步长采样：几何精度对比', fontsize=14, fontweight='bold')
    
    # 获取唯一的帧数
    frame_counts = sorted(df['frame_count'].unique())
    baseline_cds = []
    l2_cds = []
    
    for fc in frame_counts:
        baseline = df[(df['frame_count'] == fc) & (df['method'] == '固定步长采样')]
        l2 = df[(df['frame_count'] == fc) & (df['method'] == 'L2范数指导')]
        
        if len(baseline) > 0:
            baseline_cds.append(baseline['cd'].values[0])
        if len(l2) > 0:
            l2_cds.append(l2['cd'].values[0])
    
    frame_counts = frame_counts[:len(baseline_cds)]  # 对齐长度
    
    # 柱状图对比
    x = np.arange(len(frame_counts))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_cds, width, label='固定步长采样(基线)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, l2_cds, width, label='L2范数指导(本文)', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('序列长度(帧数)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('CD (cm)', fontsize=11, fontweight='bold')
    ax1.set_title('Chamfer Distance 絕對值对比', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(fc)}F' for fc in frame_counts])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 改进百分比折线图
    improvements = []
    for baseline_cd, l2_cd in zip(baseline_cds, l2_cds):
        improvement = (baseline_cd - l2_cd) / baseline_cd * 100
        improvements.append(improvement)
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    ax2.plot(frame_counts, improvements, marker='o', linewidth=2.5, markersize=8, color='#27ae60', label='CD改进率')
    ax2.fill_between(frame_counts, 0, improvements, alpha=0.3, color='#27ae60')
    
    # 添加改进百分比标签
    for fc, imp in zip(frame_counts, improvements):
        ax2.text(fc, imp + 0.5, f'{imp:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('序列长度(帧数)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('CD改进率 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('L2范数方法相对改进', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticklabels([f'{int(fc)}F' for fc in frame_counts])
    
    # 颜色标注
    if any(imp > 0 for imp in improvements):
        ax2.text(0.98, 0.97, '✓ 改进', transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / 'cd_comparison_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()


def create_multiaxis_metrics_chart(df, output_dir):
    """图2: 多指标综合对比(CD/ATE/ARE)"""
    frame_counts = sorted(df['frame_count'].unique())
    n_metrics = 3
    n_frames = len(frame_counts)
    
    # 准备数据
    metrics_data = {
        'CD': {'baseline': [], 'l2': []},
        'ATE': {'baseline': [], 'l2': []},
        'ARE': {'baseline': [], 'l2': []},
    }
    
    for fc in frame_counts:
        baseline = df[(df['frame_count'] == fc) & (df['method'] == '固定步长采样')]
        l2 = df[(df['frame_count'] == fc) & (df['method'] == 'L2范数指导')]
        
        if len(baseline) > 0 and len(l2) > 0:
            metrics_data['CD']['baseline'].append(baseline['cd'].values[0])
            metrics_data['CD']['l2'].append(l2['cd'].values[0])
            metrics_data['ATE']['baseline'].append(baseline['ate'].values[0])
            metrics_data['ATE']['l2'].append(l2['ate'].values[0])
            metrics_data['ARE']['baseline'].append(baseline['are'].values[0])
            metrics_data['ARE']['l2'].append(l2['are'].values[0])
    
    # 创建3×1的子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('多指标综合评估：L2范数 vs 固定步长采样', fontsize=14, fontweight='bold', y=0.995)
    
    metric_names = ['CD (cm)', 'ATE (m)', 'ARE (度)']
    colors_baseline = '#3498db'
    colors_l2 = '#e74c3c'
    
    for idx, (ax, metric) in enumerate(zip(axes, ['CD', 'ATE', 'ARE'])):
        x = np.arange(len(frame_counts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, metrics_data[metric]['baseline'], 
                      width, label='固定步长采样', color=colors_baseline, alpha=0.8)
        bars2 = ax.bar(x + width/2, metrics_data[metric]['l2'], 
                      width, label='L2范数指导', color=colors_l2, alpha=0.8)
        
        ax.set_ylabel(metric_names[idx], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(fc)}F' for fc in frame_counts])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加改进率注解
        improvements = []
        for baseline_val, l2_val in zip(metrics_data[metric]['baseline'], metrics_data[metric]['l2']):
            # CD/ATE越低越好，ARE越低越好，所以改进定义一致
            improvement = (baseline_val - l2_val) / baseline_val * 100
            improvements.append(improvement)
        
        for i, (fc, imp) in enumerate(zip(frame_counts, improvements)):
            color = '#27ae60' if imp > 0 else '#e67e22'
            ax.text(i, max(metrics_data[metric]['baseline'][i], metrics_data[metric]['l2'][i]) * 0.5,
                   f'{imp:+.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    output_path = output_dir / 'multiaxis_metrics_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()


def create_accuracy_speed_tradeoff(df, output_dir):
    """图3: 精度-性能权衡分析(Pareto曲线)"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 提取数据
    baseline_data = df[df['method'] == '固定步长采样'].sort_values('frame_count')
    l2_data = df[df['method'] == 'L2范数指导'].sort_values('frame_count')
    
    ax.scatter(baseline_data['time_ms'], baseline_data['cd'], 
              s=200, alpha=0.7, color='#3498db', marker='o', 
              label='固定步长采样', edgecolors='black', linewidth=1.5)
    ax.scatter(l2_data['time_ms'], l2_data['cd'], 
              s=200, alpha=0.7, color='#e74c3c', marker='s', 
              label='L2范数指导', edgecolors='black', linewidth=1.5)
    
    # 连接同一帧数的点
    for fc in sorted(df['frame_count'].unique()):
        baseline = baseline_data[baseline_data['frame_count'] == fc]
        l2 = l2_data[l2_data['frame_count'] == fc]
        
        if len(baseline) > 0 and len(l2) > 0:
            ax.plot([baseline['time_ms'].values[0], l2['time_ms'].values[0]], 
                   [baseline['cd'].values[0], l2['cd'].values[0]], 
                   'k--', alpha=0.2, linewidth=1)
    
    # 添加帧数标签
    for idx, row in baseline_data.iterrows():
        ax.annotate(f"{int(row['frame_count'])}F\n基线", 
                   (row['time_ms'], row['cd']), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, alpha=0.7)
    
    for idx, row in l2_data.iterrows():
        ax.annotate(f"{int(row['frame_count'])}F\nL2", 
                   (row['time_ms'], row['cd']), 
                   xytext=(5, -10), textcoords='offset points', 
                   fontsize=9, alpha=0.7)
    
    ax.set_xlabel('推理时间 (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('几何精度 CD (cm)', fontsize=12, fontweight='bold')
    ax.set_title('精度-性能权衡分析(越左下越优)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注改进方向
    ax.annotate('改进方向\n(精度↑ 速度↑)', xy=(0.15, 0.15), xycoords='axes fraction',
               fontsize=10, ha='center', color='#27ae60', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.2))
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_speed_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()


def create_improvement_heatmap(df, output_dir):
    """图4: 改进百分比热力图"""
    # 只有一个方法对比，构建改进矩阵
    frame_counts = sorted(df['frame_count'].unique())
    
    improvements = []
    metrics = ['CD', 'ATE', 'ARE']
    
    for fc in frame_counts:
        baseline = df[(df['frame_count'] == fc) & (df['method'] == '固定步长采样')]
        l2 = df[(df['frame_count'] == fc) & (df['method'] == 'L2范数指导')]
        
        if len(baseline) > 0 and len(l2) > 0:
            cd_imp = (baseline['cd'].values[0] - l2['cd'].values[0]) / baseline['cd'].values[0] * 100
            ate_imp = (baseline['ate'].values[0] - l2['ate'].values[0]) / baseline['ate'].values[0] * 100
            are_imp = (baseline['are'].values[0] - l2['are'].values[0]) / baseline['are'].values[0] * 100
            improvements.append([cd_imp, ate_imp, are_imp])
    
    improvements = np.array(improvements).T  # [3, n_frames]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    ax.set_xticks(np.arange(len(frame_counts)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([f'{int(fc)}F' for fc in frame_counts])
    ax.set_yticklabels(metrics)
    
    # 添加数值
    for i in range(len(metrics)):
        for j in range(len(frame_counts)):
            text = ax.text(j, i, f'{improvements[i, j]:+.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    ax.set_title('L2范数方法相对改进百分比热力图\n(绿色=改进, 红色=恶化)', fontsize=12, fontweight='bold')
    ax.set_xlabel('序列长度', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('改进率 (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()


def create_summary_table(df, output_dir):
    """表格1: 详细统计表格"""
    # 创建汇总表格
    frame_counts = sorted(df['frame_count'].unique())
    
    summary_rows = []
    
    for fc in frame_counts:
        baseline = df[(df['frame_count'] == fc) & (df['method'] == '固定步长采样')]
        l2 = df[(df['frame_count'] == fc) & (df['method'] == 'L2范数指导')]
        
        if len(baseline) > 0 and len(l2) > 0:
            b = baseline.iloc[0]
            l = l2.iloc[0]
            
            cd_imp = (b['cd'] - l['cd']) / b['cd'] * 100
            ate_imp = (b['ate'] - l['ate']) / b['ate'] * 100
            are_imp = (b['are'] - l['are']) / b['are'] * 100
            time_change = (l['time_ms'] - b['time_ms']) / b['time_ms'] * 100
            
            summary_rows.append({
                '序列长度': f'{int(fc)}F',
                '方法': '基线',
                'CD(cm)': f"{b['cd']:.4f}",
                'ATE(m)': f"{b['ate']:.4f}",
                'ARE(°)': f"{b['are']:.2f}",
                '时间(ms)': f"{b['time_ms']:.0f}",
            })
            
            summary_rows.append({
                '序列长度': f'{int(fc)}F',
                '方法': 'L2指导',
                'CD(cm)': f"{l['cd']:.4f}",
                'ATE(m)': f"{l['ate']:.4f}",
                'ARE(°)': f"{l['are']:.2f}",
                '时间(ms)': f"{l['time_ms']:.0f}",
            })
            
            summary_rows.append({
                '序列长度': f'{int(fc)}F',
                '方法': '改进',
                'CD(cm)': f"{cd_imp:+.2f}%",
                'ATE(m)': f"{ate_imp:+.2f}%",
                'ARE(°)': f"{are_imp:+.2f}%",
                '时间(ms)': f"{time_change:+.2f}%",
            })
            
            summary_rows.append({
                '序列长度': '',
                '方法': '',
                'CD(cm)': '',
                'ATE(m)': '',
                'ARE(°)': '',
                '时间(ms)': '',
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # 保存为CSV
    csv_path = output_dir / 'comprehensive_results_summary.csv'
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {csv_path}")
    
    # 创建美化的表格图像
    fig, ax = plt.subplots(figsize=(12, len(summary_rows)*0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 美化表头
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 美化数据行
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 4 == 0:  # 空行
                table[(i, j)].set_facecolor('#ecf0f1')
            elif summary_df.iloc[i-1, j] == '改进':
                table[(i, j)].set_facecolor('#d5f4e6')
                table[(i, j)].set_text_props(weight='bold', color='#27ae60')
            elif summary_df.iloc[i-1, j] == '基线':
                table[(i, j)].set_facecolor('#e8f8f5')
            elif summary_df.iloc[i-1, j] == 'L2指导':
                table[(i, j)].set_facecolor('#fadbd8')
    
    plt.title('详细测试结果统计表', fontsize=13, fontweight='bold', pad=20)
    
    table_path = output_dir / 'summary_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {table_path}")
    plt.close()


def generate_analysis_report(df, output_dir):
    """生成详细分析报告"""
    report = """
================================================================================
        L2范数Token划分有效性验证 - 综合实验分析报告
================================================================================

【实验目标】
  验证基于L2范数的几何一致性Token划分策略是否相比固定步长采样更优

【关键参数】
  - 合并率: 30% (merge_ratio = 0.30)
  - 保护比: 10% (Protected tokens)
  - 目标比: 40% (Dst tokens for frame1+)
  - 源比: 50% (Src tokens for merging)
  - 评估数据集: ScanNet (2个场景)

【实验结果摘要】
"""
    
    frame_counts = sorted(df['frame_count'].unique())
    
    total_cd_improvement = 0
    total_ate_improvement = 0
    total_are_improvement = 0
    total_time_change = 0
    count = 0
    
    report += "\n【逐帧分析】\n"
    report += "-" * 80 + "\n"
    
    for fc in frame_counts:
        baseline = df[(df['frame_count'] == fc) & (df['method'] == '固定步长采样')]
        l2 = df[(df['frame_count'] == fc) & (df['method'] == 'L2范数指导')]
        
        if len(baseline) > 0 and len(l2) > 0:
            b = baseline.iloc[0]
            l = l2.iloc[0]
            
            cd_imp = (b['cd'] - l['cd']) / b['cd'] * 100
            ate_imp = (b['ate'] - l['ate']) / b['ate'] * 100
            are_imp = (b['are'] - l['are']) / b['are'] * 100
            time_change = (l['time_ms'] - b['time_ms']) / b['time_ms'] * 100
            
            total_cd_improvement += cd_imp
            total_ate_improvement += ate_imp
            total_are_improvement += are_imp
            total_time_change += time_change
            count += 1
            
            report += f"\n{int(fc)}帧测试结果:\n"
            report += f"  基线: CD={b['cd']:.4f}cm, ATE={b['ate']:.4f}m, ARE={b['are']:.2f}°, 时间={b['time_ms']:.0f}ms\n"
            report += f"  L2法: CD={l['cd']:.4f}cm, ATE={l['ate']:.4f}m, ARE={l['are']:.2f}°, 时间={l['time_ms']:.0f}ms\n"
            report += f"  改进: CD {cd_imp:+.2f}%, ATE {ate_imp:+.2f}%, ARE {are_imp:+.2f}%, 时间 {time_change:+.2f}%\n"
            
            if cd_imp > 0:
                report += f"  🟢 L2方法在几何精度上改进了{cd_imp:.2f}%\n"
            else:
                report += f"  🔴 L2方法在几何精度上恶化了{abs(cd_imp):.2f}%\n"
    
    if count > 0:
        avg_cd = total_cd_improvement / count
        avg_ate = total_ate_improvement / count
        avg_are = total_are_improvement / count
        avg_time = total_time_change / count
        
        report += "\n" + "-" * 80 + "\n"
        report += f"\n【整体平均指标】\n"
        report += f"  CD改进: {avg_cd:+.2f}%  {'✓' if avg_cd > 0 else '✗'}\n"
        report += f"  ATE改进: {avg_ate:+.2f}%  {'✓' if avg_ate > 0 else '✗'}\n"
        report += f"  ARE改进: {avg_are:+.2f}%  {'✓' if avg_are > 0 else '✗'}\n"
        report += f"  推理时间变化: {avg_time:+.2f}%\n"
    
    report += "\n" + "=" * 80 + "\n"
    report += "\n【结论】\n"
    
    if avg_cd > 0:
        report += f"""
✅ L2范数Token划分方法成功改进几何精度

通过使用L2范数来识别和保护高重要性的Token，我们实现了：
  • 几何精度提升: {avg_cd:+.2f}% (Chamfer Distance减少)
  • 姿态精度提升: {avg_ate:+.2f}% (ATE减少)
  • 旋转精度提升: {avg_are:+.2f}% (ARE减少)
  • 推理成本变化: {avg_time:+.2f}%

建议：在精度优先的应用场景中部署该方法
"""
    else:
        report += f"""
⚠️  L2范数方法在平均性能上未达预期

需要进一步调查：
  1. L2范数度量是否真正捕捉了几何重要性
  2. 参数配置(P=10%, D=40%)是否与合并率30%兼容
  3. 是否需要进行参数网格搜索优化

建议：进行参数敏感性分析和消融实验
"""
    
    report += "\n" + "=" * 80 + "\n"
    report += "\n【后续工作】\n"
    report += """
1. 扩大评估规模(5-10个场景而非2个)以提高统计显著性
2. 进行参数网格搜索优化(P/D/S组合)
3. 对比其他Token importance指标(方差、注意力权重等)
4. 可视化Token划分结果，理解设计的实际效果
5. 在长序列(200+帧)上验证扩展性
"""
    
    # 保存报告
    report_path = output_dir / 'comprehensive_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ 已保存: {report_path}")
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


def main():
    """主函数"""
    setup_chinese_font()
    
    # 输入输出路径
    csv_path = Path('tests/tests_result/l2_partition_effectiveness_v2/partition_comparison_results_v2.csv')
    output_dir = Path('tests/tests_result/l2_partition_effectiveness_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查CSV是否存在
    if not csv_path.exists():
        print(f"❌ 错误: 找不到结果文件 {csv_path}")
        print("请先运行: python tests/test_l2_partition_effectiveness_v2.py")
        return
    
    # 加载数据
    df = load_results(csv_path)
    
    # 检查数据完整性
    frame_counts = sorted(df['frame_count'].unique())
    print(f"\n  发现 {len(frame_counts)} 种帧长: {frame_counts}")
    
    # 生成可视化
    print("\n正在生成可视化图表...")
    
    try:
        create_cd_comparison_chart(df, output_dir)
    except Exception as e:
        print(f"⚠️  CD对比图表生成失败: {e}")
    
    try:
        create_multiaxis_metrics_chart(df, output_dir)
    except Exception as e:
        print(f"⚠️  多轴指标图表生成失败: {e}")
    
    try:
        create_accuracy_speed_tradeoff(df, output_dir)
    except Exception as e:
        print(f"⚠️  精度-性能权衡图生成失败: {e}")
    
    try:
        create_improvement_heatmap(df, output_dir)
    except Exception as e:
        print(f"⚠️  热力图生成失败: {e}")
    
    try:
        create_summary_table(df, output_dir)
    except Exception as e:
        print(f"⚠️  统计表格生成失败: {e}")
    
    # 生成分析报告
    print("\n正在生成分析报告...")
    try:
        generate_analysis_report(df, output_dir)
    except Exception as e:
        print(f"⚠️  分析报告生成失败: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 所有可视化完成！")
    print(f"📊 输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
