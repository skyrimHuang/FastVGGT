"""
根据公式
r_l=min(r_max, max(r_min, C_base(l) * (1 + beta * max(0, S - S_base))))
绘制三维网格图：x轴为层号l，y轴为序列长度S，z轴为r。

使用方式示例：
python tests/plot_r_surface_by_beta.py --beta 0.003
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from scipy.interpolate import make_interp_spline


def setup_chinese_font() -> None:
    """全局设置中文字体，避免中文乱码。"""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    available_font_names = {font.name for font in fm.fontManager.ttflist}
    selected_fonts = [name for name in preferred_fonts if name in available_font_names]
    if not selected_fonts:
        selected_fonts = ["DejaVu Sans"]

    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def parse_float_list(text: str, expected_len: int) -> list[float]:
    """解析逗号分隔的浮点数列表。"""
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"需要 {expected_len} 个数值，当前得到 {len(values)} 个：{text}")
    return values


def build_cbase_curve(
    anchor_layers: np.ndarray,
    anchor_values: np.ndarray,
    l_samples: np.ndarray,
) -> np.ndarray:
    """用二次样条插值计算任意层l对应的C_base(l)。"""
    spline = make_interp_spline(anchor_layers, anchor_values, k=2)
    cbase = spline(l_samples)
    return cbase


def compute_r_grid(
    l_grid: np.ndarray,
    s_grid: np.ndarray,
    cbase_grid: np.ndarray,
    beta: float,
    s_base: float,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """按给定公式计算r网格。"""
    growth = 1.0 + beta * np.maximum(0.0, s_grid - s_base)
    raw_r = cbase_grid * growth
    clipped_r = np.clip(raw_r, r_min, r_max)
    return clipped_r


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制l-S-r三维网格图（支持输入beta）")
    parser.add_argument("--beta", type=float, required=True, help="超参数beta（你描述中的b）")
    parser.add_argument("--s_base", type=float, default=100.0, help="S_base，默认100")
    parser.add_argument("--r_min", type=float, default=0.0, help="r最小值")
    parser.add_argument("--r_max", type=float, default=0.9, help="r最大值")
    parser.add_argument("--l_min", type=float, default=0, help="层号下界")
    parser.add_argument("--l_max", type=float, default=23, help="层号上界")
    parser.add_argument("--s_min", type=float, default=5.0, help="S下界")
    parser.add_argument("--s_max", type=float, default=250.0, help="S上界")
    parser.add_argument("--num_l", type=int, default=96, help="l方向均匀采样数量")
    parser.add_argument("--num_s", type=int, default=96, help="S方向均匀采样数量")
    parser.add_argument(
        "--anchor_values",
        type=str,
        default="0.4,0.3,0.8,0.9",
        help="四个锚点值，逗号分隔，默认0.4,0.3,0.8,0.9",
    )
    parser.add_argument(
        "--anchor_layers",
        type=str,
        default="0,8,16,23",
        help="四个锚点对应层号，逗号分隔，默认0,8,16,23",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tests/tests_result/pareto_analysis/figures"),
        help="输出目录",
    )

    args = parser.parse_args()

    setup_chinese_font()

    anchor_values = np.array(parse_float_list(args.anchor_values, expected_len=4), dtype=np.float64)
    anchor_layers = np.array(parse_float_list(args.anchor_layers, expected_len=4), dtype=np.float64)

    if not np.all(np.diff(anchor_layers) > 0):
        raise ValueError("anchor_layers 必须严格递增，例如 0,8,16,23")

    l_samples = np.linspace(args.l_min, args.l_max, args.num_l)
    s_samples = np.linspace(args.s_min, args.s_max, args.num_s)

    cbase_values = build_cbase_curve(anchor_layers, anchor_values, l_samples)

    l_grid, s_grid = np.meshgrid(l_samples, s_samples)
    cbase_grid = np.tile(cbase_values, (args.num_s, 1))

    r_grid = compute_r_grid(
        l_grid=l_grid,
        s_grid=s_grid,
        cbase_grid=cbase_grid,
        beta=args.beta,
        s_base=args.s_base,
        r_min=args.r_min,
        r_max=args.r_max,
    )

    fig = plt.figure(figsize=(13, 10), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # 曲面 + 网格线，形成“均匀采样的一张网”视觉效果
    surface = ax.plot_surface(
        l_grid,
        s_grid,
        r_grid,
        cmap="viridis",
        linewidth=0.2,
        edgecolor=(0.4, 0.4, 0.4, 0.55),
        antialiased=True,
        alpha=0.9,
    )
    ax.plot_wireframe(
        l_grid,
        s_grid,
        r_grid,
        rstride=max(1, args.num_s // 20),
        cstride=max(1, args.num_l // 20),
        color=(0.1, 0.1, 0.1, 0.35),
        linewidth=0.35,
    )

    ax.set_xlabel("层号 l", labelpad=12, fontsize=12, fontweight="bold")
    ax.set_ylabel("序列长度 S", labelpad=12, fontsize=12, fontweight="bold")
    ax.set_zlabel("合并率 r", labelpad=12, fontsize=12, fontweight="bold")

    # 设置层号轴为整数刻度
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    ax.set_title(
        f"自适应合并率三维图（β={args.beta}）\n"
        f"锚点值={list(anchor_values)}，S_base={args.s_base}",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    color_bar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=20, pad=0.08)
    color_bar.set_label("r 值大小", fontsize=11)

    ax.view_init(elev=28, azim=-132)
    ax.set_xlim(args.l_min, args.l_max)
    ax.set_ylim(args.s_min, args.s_max)
    ax.set_zlim(args.r_min, args.r_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    beta_name = str(args.beta).replace("-", "m").replace(".", "p")
    output_path = args.output_dir / f"r_surface_beta_{beta_name}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print("=" * 80)
    print("三维图生成完成")
    print("=" * 80)
    print(f"输出文件: {output_path}")
    print(f"β: {args.beta}")
    print(f"l范围: [{args.l_min}, {args.l_max}], 采样数: {args.num_l}")
    print(f"S范围: [{args.s_min}, {args.s_max}], 采样数: {args.num_s}")
    print(f"r范围(裁剪后): [{r_grid.min():.4f}, {r_grid.max():.4f}]")


if __name__ == "__main__":
    main()
