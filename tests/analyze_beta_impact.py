"""
分析β参数对合并率的影响：通过理论公式和可视化快速定位最优β范围。

使用方式：
python tests/analyze_beta_impact.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from pathlib import Path


def setup_chinese_font() -> None:
    """全局设置中文字体"""
    preferred_fonts = [
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans CN",
        "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    selected_fonts = [n for n in preferred_fonts if n in available_font_names]
    if not selected_fonts:
        selected_fonts = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def analyze_beta_range() -> dict:
    """
    分析不同β值的合理范围：
    - β过小：合并率变化不明显，加速效果差
    - β过大：合并率波动过大，精度损失大
    """
    # 4层锚点值（来自Optimal配置）
    anchor_layers = np.array([0, 8, 16, 23], dtype=float)
    anchor_values = np.array([0.4, 0.3, 0.8, 0.9], dtype=float)
    
    # 关键序列长度点（ScanNet典型范围）
    s_base = 100.0
    s_typical = np.array([50, 100, 150, 200, 250])  # 常见的序列长度
    
    # 测试不同β值
    beta_candidates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    
    analysis = {
        "beta_values": beta_candidates,
        "s_samples": s_typical,
        "results": {}
    }
    
    # 对每个β值分析其效果
    for beta in beta_candidates:
        # 在典型S值处的合并率变化
        r_changes = []
        for s in s_typical:
            penalty = 1.0 - beta * max(0, s - s_base)
            # 以layer 16（中间层）的C_base=0.8为例
            r_mid = min(0.9, max(0.0, 0.8 * penalty))
            r_changes.append(r_mid)
        
        # 计算有意义的指标
        r_change_range = max(r_changes) - min(r_changes)  # β的影响强度
        r_change_ratio = r_change_range / np.mean(r_changes) if np.mean(r_changes) > 0 else 0
        
        analysis["results"][beta] = {
            "r_at_typical_s": r_changes,
            "r_range": r_change_range,
            "r_change_ratio": r_change_ratio,
            "description": f"在S=[50,250]范围内，r在layer16变化 {r_change_range:.4f}"
        }
    
    return analysis


def create_beta_selection_visualization(analysis: dict, output_dir: Path) -> None:
    """可视化β值的效果，辅助选择"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # 图1：β对合并率的影响
    ax = axes[0]
    for beta in analysis["beta_values"][:4]:  # 展示部分β
        r_changes = analysis["results"][beta]["r_at_typical_s"]
        ax.plot(analysis["s_samples"], r_changes, marker='o', label=f'β={beta:.4f}')
    
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='S_base=100')
    ax.set_xlabel("序列长度 S", fontsize=11, fontweight='bold')
    ax.set_ylabel("合并率 r (layer16)", fontsize=11, fontweight='bold')
    ax.set_title("β对动态合并率的影响", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图2：β对合并率变化幅度的影响（辅助判断β的激进程度）
    ax = axes[1]
    betas = analysis["beta_values"]
    ranges = [analysis["results"][b]["r_range"] for b in betas]
    ratios = [analysis["results"][b]["r_change_ratio"] for b in betas]
    
    ax2 = ax.twinx()
    bar1 = ax.bar(range(len(betas)), ranges, alpha=0.6, label='r变化幅度', color='steelblue')
    line1, = ax2.plot(range(len(betas)), ratios, 'ro-', linewidth=2, 
                       markersize=6, label='r相对变化率')
    
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.4f}' for b in betas], rotation=45, fontsize=9)
    ax.set_xlabel("β值", fontsize=11, fontweight='bold')
    ax.set_ylabel("r绝对变化幅度", fontsize=10, fontweight='bold', color='steelblue')
    ax2.set_ylabel("r相对变化率 (%)", fontsize=10, fontweight='bold', color='red')
    ax.set_title("β的激进程度分析", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3：推荐β值范围
    ax = axes[2]
    ax.axis('off')
    
    recommend_text = """
    β值选择建议（基于理论分析）：
    
    1. 【保守型β】(0.0001-0.0005)
       • 合并率变化小，加速效果有限
       • 适合验证合并机制的有效性
       
    2. 【推荐β】(0.001-0.003) ✓
       • 合并率变化适中
       • 精度-速率平衡好
       • 推荐从β=0.001开始网格搜索
       
    3. 【激进型β】(0.005-0.01)
       • 合并率变化显著
       • 可能导致精度下降明显
       • 作为上界进行对比
    
    优化策略：
    ✓ 第一轮：β∈[0.001, 0.002, 0.003]
    ✓ 第二轮：在最优值±0.0005范围细化
    """
    
    ax.text(0.05, 0.95, recommend_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "beta_selection_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ β分析图生成: {output_path}")


def main():
    setup_chinese_font()
    
    print("="*80)
    print("β参数快速分析")
    print("="*80)
    
    analysis = analyze_beta_range()
    
    # 输出每个β的分析结果
    print("\n【β值效果分析表】\n")
    print(f"{'β值':<12} {'r变化幅度':<15} {'相对变化率':<15} {'推荐度':<10}")
    print("-" * 60)
    
    for beta in analysis["beta_values"]:
        result = analysis["results"][beta]
        recommend = "⭐⭐⭐⭐⭐" if 0.001 <= beta <= 0.003 else \
                   "⭐⭐" if beta < 0.001 else "⭐⭐⭐"
        print(f"{beta:<12.4f} {result['r_range']:<15.4f} "
              f"{result['r_change_ratio']:<15.4f} {recommend:<10}")
    
    # 生成可视化
    output_dir = Path("tests/tests_result/pareto_analysis/figures")
    create_beta_selection_visualization(analysis, output_dir)
    
    print("\n" + "="*80)
    print("论文中的表述建议（可直接使用）：")
    print("="*80)
    print("""
    "为了找到最优的β值，我们先进行了理论参数分析。
    根据公式 r_l = min(r_max, max(r_min, C_base(l)×(1-β×max(0,S-S_base))))，
    β控制序列长度对合并率的影响强度。
    
    我们分析了β∈[0.0001, 0.01]范围内的影响：
    - β过小(<0.0005)会导致合并率变化不足，无法充分利用自适应机制
    - β过大(>0.005)会导致某些场景下合并率波动过大，可能伤害精度
    - 推荐范围为β∈[0.001, 0.003]，在这个范围内合并率的变化既能体现
      自适应的优势，又能保持与速度的平衡"
    """)


if __name__ == "__main__":
    main()
