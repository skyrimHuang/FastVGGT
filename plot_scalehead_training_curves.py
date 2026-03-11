#!/usr/bin/env python3
"""绘制ScaleHead训练过程多曲线图。"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def setup_chinese_font() -> None:
    """全局设置中文字体，避免乱码。"""
    cjk_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        font_manager.fontManager.addfont(cjk_font_path)
    except Exception:
        pass

    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 10


def plot_training_curves(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)

    epochs = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy()
    val_mean_error_pct = df["val_mean_error"].to_numpy() * 100.0
    val_median_error_pct = df["val_median_error"].to_numpy() * 100.0
    lr = df["learning_rate"].to_numpy()
    delta_005 = df["delta_0.05"].to_numpy() * 100.0
    delta_010 = df["delta_0.10"].to_numpy() * 100.0
    delta_015 = df["delta_0.15"].to_numpy() * 100.0

    best_idx = int(np.argmin(val_mean_error_pct))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_mean_error_pct[best_idx])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="tab:blue", linewidth=2.0, marker="o", markersize=3, label="Train Loss")
    ax.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.6, label=f"最优Epoch={best_epoch}")
    ax.set_title("(a) 训练损失曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(epochs, val_mean_error_pct, color="tab:red", linewidth=2.0, marker="o", markersize=3, label="Val Mean Error")
    ax.plot(epochs, val_median_error_pct, color="tab:orange", linewidth=1.8, marker="s", markersize=3, label="Val Median Error")
    ax.scatter([best_epoch], [best_val], color="black", s=40, zorder=5)
    ax.annotate(f"最优: {best_val:.2f}% (ep {best_epoch})", xy=(best_epoch, best_val),
                xytext=(best_epoch + 1, best_val + 2.2), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"))
    ax.set_title("(b) 验证集尺度误差")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(epochs, delta_005, color="tab:green", linewidth=1.8, marker="o", markersize=3, label="δ<0.05")
    ax.plot(epochs, delta_010, color="tab:purple", linewidth=1.8, marker="s", markersize=3, label="δ<0.10")
    ax.plot(epochs, delta_015, color="tab:brown", linewidth=2.0, marker="^", markersize=3, label="δ<0.15")
    ax.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_title("(c) 阈值达标率曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("达标率 (%)")
    ax.set_ylim(0, max(70, np.max(delta_015) + 5))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(epochs, lr, color="tab:cyan", linewidth=2.0, label="Learning Rate")
    ax2.plot(epochs, val_mean_error_pct, color="tab:red", linewidth=1.8, linestyle="--", label="Val Mean Error (%)")
    ax.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_title("(d) 学习率与验证误差")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate", color="tab:cyan")
    ax2.set_ylabel("Val Error (%)", color="tab:red")
    ax.grid(alpha=0.3)

    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=9)

    fig.suptitle("ScaleHead训练过程可视化（KITTI v2, Early Stopping@32 epochs）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✓ 图表已保存: {output_path}")
    print(f"最优Epoch: {best_epoch}")
    print(f"最优验证误差: {best_val:.4f}%")
    print(f"最终Epoch: {int(epochs[-1])}")
    print(f"最终验证误差: {float(val_mean_error_pct[-1]):.4f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制ScaleHead训练过程多曲线图")
    parser.add_argument("--csv", type=Path, required=True, help="training_history.csv路径")
    parser.add_argument("--output", type=Path, required=True, help="图像输出路径")
    args = parser.parse_args()

    setup_chinese_font()
    plot_training_curves(args.csv, args.output)


if __name__ == "__main__":
    main()
