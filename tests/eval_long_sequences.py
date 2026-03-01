"""
超长序列评估 —— OOM边界分析

对长视频序列（100-1000帧）进行评估，展示：
  1. 关键帧过滤的必要性：无过滤在长序列上快速OOM
  2. OOM边界曲线：展示不同阈值下的OOM点
  3. 显存扩展性：过滤方案在长序列上的线性增长
  4. 增量结果保存：中途OOM不丢失已有数据

使用方式:
  python tests/eval_long_sequences.py \\
    --dataset_type 7scenes \\
    --data_dir /home/hba/Documents/Dataset/7_scenes \\
    --sequence_lengths 50,100,200,500,1000 \\
    --thresholds 0.1,0.3,0.5,0.7 \\
    --num_samples 1 \\
    --output_dir ./tests/eval_long_seq
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# 配置中文字体（与main eval脚本同）
def setup_chinese_font():
    import glob
    cjk_font_patterns = [
        "/usr/share/fonts/**/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansSC-Regular.otf",
        "/usr/share/fonts/**/wqy-microhei.ttc",
        "/usr/share/fonts/**/wqy-zenhei.ttc",
    ]
    font_path = None
    for pattern in cjk_font_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            font_path = matches[0]
            break

    if font_path:
        font_manager.fontManager.addfont(font_path)
        fp = font_manager.FontProperties(fname=font_path)
        font_name = fp.get_name()
        print(f"✓ 使用中文字体: {font_name}")
    else:
        font_name = "SimHei"
        fp = font_manager.FontProperties(family=font_name)
        print(f"⚠ 未找到系统CJK字体")

    rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial"]
    rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")
    return fp

FONT_PROP = setup_chinese_font()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.keyframe_filter import KeyframeFilter


def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate Long Sequences with OOM Tracking")
    parser.add_argument("--ckpt_path", type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--dataset_type", choices=["7scenes", "scannet"], default="7scenes")
    parser.add_argument("--data_dir", type=str,
        default="/home/hba/Documents/Dataset/7_scenes")
    parser.add_argument("--output_dir", type=str,
        default="./tests/eval_long_seq")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sequence_lengths", type=str,
        default="100,200,500,1000", help="长序列长度列表")
    parser.add_argument("--thresholds", type=str,
        default="0.1,0.3,0.5,0.7", help="关键帧阈值列表")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_7scenes_data(data_dir: str, num_frames: int, resolution: Tuple[int, int] = (518, 392)) -> torch.Tensor:
    """加载7Scenes数据 - 返回单个样本"""
    sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
    from data import SevenScenes
    
    dataset = SevenScenes(
        split="test",
        ROOT=data_dir,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=1,
    )
    
    views = dataset[0]
    actual_frames = min(num_frames, len(views))
    selected_views = views[:actual_frames]
    
    imgs = torch.stack([v["img"] for v in selected_views])
    imgs = (imgs + 1.0) / 2.0  # [-1, 1] → [0, 1]
    
    return imgs.unsqueeze(0)  # [1, T, 3, H, W]


def load_scannet_data(data_dir: str, num_frames: int) -> torch.Tensor:
    """加载ScanNet数据"""
    from vggt.utils.eval_utils import get_sorted_image_paths, load_images_rgb, get_vgg_input_imgs
    
    data_dir = Path(data_dir)
    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)])
    
    scene = scenes[0]
    scene_dir = data_dir / scene
    images_dir = scene_dir / "color"
    
    image_paths = get_sorted_image_paths(images_dir)
    actual_frames = min(num_frames, len(image_paths))
    selected_paths = image_paths[:actual_frames]
    
    images = load_images_rgb(selected_paths)
    images_array = np.stack(images)
    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    
    return vgg_input


def try_evaluate_no_filter(model: VGGT, images: torch.Tensor, device: str) -> Dict:
    """
    尝试运行无过滤的推理。
    
    返回:
        dict: {
            "success": bool,
            "time": float or None,
            "memory": float or None,
            "error": str or None
        }
    """
    images = images.to(device, dtype=torch.float32)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                t0 = time.time()
                _ = model(images)
                torch.cuda.synchronize(device)
                elapsed = time.time() - t0
        
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
        return {
            "success": True,
            "time": elapsed,
            "memory": peak_mem,
            "error": None,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "success": False,
            "time": None,
            "memory": None,
            "error": "OOM",
        }
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "success": False,
            "time": None,
            "memory": None,
            "error": str(type(e).__name__),
        }


def try_evaluate_with_filter(model: VGGT, images: torch.Tensor, threshold: float, device: str) -> Dict:
    """
    尝试运行有过滤的推理。
    
    返回:
        dict: {
            "success": bool,
            "compression_ratio": float or None,
            "time_filter": float or None,
            "time_inference": float or None,
            "time_total": float or None,
            "memory": float or None,
            "error": str or None,
        }
    """
    images = images.to(device, dtype=torch.float32)
    aggregator = model.aggregator
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=threshold)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            # 特征提取
            with torch.cuda.amp.autocast(dtype=torch.float16):
                t0 = time.time()
                filter_result = filter_model(images)
                torch.cuda.synchronize(device)
                t_filter = time.time() - t0
            
            filtered_images = filter_result["filtered_images"].to(device, dtype=torch.float32)
            patch_tokens = filter_result["patch_tokens"].to(device, dtype=torch.bfloat16)
            
            # 推理
            with torch.cuda.amp.autocast(dtype=torch.float16):
                t0 = time.time()
                _ = model(filtered_images, precomputed_patch_tokens=patch_tokens)
                torch.cuda.synchronize(device)
                t_inference = time.time() - t0
        
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
        compression_ratio = filter_result["stats"]["compression_ratio"]
        
        return {
            "success": True,
            "compression_ratio": compression_ratio,
            "time_filter": t_filter,
            "time_inference": t_inference,
            "time_total": t_filter + t_inference,
            "memory": peak_mem,
            "error": None,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "success": False,
            "compression_ratio": None,
            "time_filter": None,
            "time_inference": None,
            "time_total": None,
            "memory": None,
            "error": "OOM",
        }
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "success": False,
            "compression_ratio": None,
            "time_filter": None,
            "time_inference": None,
            "time_total": None,
            "memory": None,
            "error": str(type(e).__name__),
        }


def plot_oom_boundary(results_df: pd.DataFrame, output_path: str):
    """绘制OOM边界：序列长度 vs 是否成功"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: 无过滤的OOM曲线
    ax = axes[0, 0]
    no_filter_data = results_df[results_df["method"] == "no_filter"].copy()
    success_by_len = no_filter_data.groupby("sequence_length")["success"].mean() * 100
    
    ax.bar(success_by_len.index, success_by_len.values, color="red", alpha=0.7, label="成功率")
    for idx, val in success_by_len.items():
        ax.text(idx, val + 2, f"{val:.0f}%", ha="center", fontsize=9)
    ax.set_xlabel("序列长度", fontproperties=FONT_PROP)
    ax.set_ylabel("成功率 (%)", fontproperties=FONT_PROP)
    ax.set_title("无过滤：OOM风险", fontproperties=FONT_PROP)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis="y")
    
    # 右上: 不同阈值下的OOM曲线
    ax = axes[0, 1]
    filter_data = results_df[results_df["method"] == "filter"].copy()
    for threshold in sorted(filter_data["threshold"].unique()):
        subset = filter_data[filter_data["threshold"] == threshold]
        success_by_len = subset.groupby("sequence_length")["success"].mean() * 100
        ax.plot(success_by_len.index, success_by_len.values, marker="o", 
               label=f"τ={threshold}", linewidth=2)
    
    ax.set_xlabel("序列长度", fontproperties=FONT_PROP)
    ax.set_ylabel("成功率 (%)", fontproperties=FONT_PROP)
    ax.set_title("过滤方案：阈值vs OOM风险", fontproperties=FONT_PROP)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=9, prop=FONT_PROP)
    ax.grid(True, alpha=0.3)
    
    # 左下: 成功情况下的显存对比
    ax = axes[1, 0]
    no_filter_success = no_filter_data[no_filter_data["success"] == True]
    filter_success = filter_data[filter_data["success"] == True]
    
    seq_lens = sorted(filter_success["sequence_length"].unique())
    no_filter_mem = [no_filter_success[no_filter_success["sequence_length"] == s]["memory"].mean() 
                     for s in seq_lens]
    filter_mem = [filter_success[filter_success["sequence_length"] == s]["memory"].mean() 
                  for s in seq_lens]
    
    x = np.arange(len(seq_lens))
    width = 0.35
    ax.bar(x - width/2, no_filter_mem, width, label="无过滤", alpha=0.8)
    ax.bar(x + width/2, filter_mem, width, label="过滤(τ=0.3)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.set_xlabel("序列长度", fontproperties=FONT_PROP)
    ax.set_ylabel("峰值显存 (MB)", fontproperties=FONT_PROP)
    ax.set_title("显存使用对比（成功情况）", fontproperties=FONT_PROP)
    ax.legend(fontsize=10, prop=FONT_PROP)
    ax.grid(True, alpha=0.3, axis="y")
    
    # 右下: 显存与序列长度的扩展性
    ax = axes[1, 1]
    filter_success_by_threshold = {}
    for threshold in sorted(filter_data["threshold"].unique()):
        subset = filter_data[(filter_data["threshold"] == threshold) & (filter_data["success"] == True)]
        memory_by_len = subset.groupby("sequence_length")["memory"].mean()
        ax.plot(memory_by_len.index, memory_by_len.values, marker="s", 
               label=f"τ={threshold}", linewidth=2)
    
    ax.set_xlabel("序列长度", fontproperties=FONT_PROP)
    ax.set_ylabel("峰值显存 (MB)", fontproperties=FONT_PROP)
    ax.set_title("过滤方案：显存扩展性", fontproperties=FONT_PROP)
    ax.legend(fontsize=9, prop=FONT_PROP)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存OOM边界图: {output_path}")


def plot_oom_heatmap(results_df: pd.DataFrame, method: str, output_path: str):
    """绘制OOM热力图：序列长度 x 阈值（仅限过滤方案）"""
    if method == "no_filter":
        print("⚠ 无过滤方案无需阈值热力图")
        return
    
    subset = results_df[results_df["method"] == "filter"].copy()
    
    # 准备热力图数据：1=成功，0=OOM
    pivot_data = subset.pivot_table(
        index="threshold",
        columns="sequence_length",
        values="success",
        aggfunc="mean"
    ) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels([f"{t:.2f}" for t in pivot_data.index])
    
    ax.set_xlabel("序列长度", fontsize=12, fontproperties=FONT_PROP)
    ax.set_ylabel("关键帧阈值", fontsize=12, fontproperties=FONT_PROP)
    ax.set_title("过滤方案OOM热力图（%成功）", fontsize=14, fontproperties=FONT_PROP)
    
    # 添加百分比标签
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            color = "black" if val > 50 else "white"
            text = ax.text(j, i, f"{val:.0f}%", ha="center", va="center", 
                          color=color, fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("成功率 (%)", fontproperties=FONT_PROP)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存OOM热力图: {output_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print("超长序列评估 —— OOM边界分析")
    print("=" * 70)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print(f"\n[1/3] 加载模型...")
    model = VGGT(
        merging=25,
        merge_ratio=0.0,
        enable_point=True,
        enable_depth=True,
        enable_camera=True
    )
    
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"✓ 加载检查点: {args.ckpt_path}")
    
    model = model.to(device).eval()
    
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    
    results = []
    csv_path = os.path.join(args.output_dir, "eval_long_seq_results.csv")
    
    print(f"\n[2/3] 长序列评估...（增量保存至 {csv_path}）")
    
    for seq_len in tqdm(sequence_lengths, desc="序列长度"):
        try:
            if args.dataset_type == "7scenes":
                images = load_7scenes_data(args.data_dir, seq_len, tuple(args.resolution))
            else:
                images = load_scannet_data(args.data_dir, seq_len)
            
            if images.shape[0] == 0:
                print(f"\n⚠ 无法加载序列长度 {seq_len} 的数据")
                continue
            
            images = images[:1]
            print(f"\n序列长度 {seq_len}: 数据形状 {images.shape}")
            
            # 评估无过滤方案
            print(f"  → 无过滤...", end="", flush=True)
            no_filter_result = try_evaluate_no_filter(model, images, device)
            print(f" {'✓' if no_filter_result['success'] else '✗ OOM'}")
            
            results.append({
                "sequence_length": seq_len,
                "method": "no_filter",
                "threshold": None,
                "success": no_filter_result["success"],
                "time": no_filter_result["time"],
                "memory": no_filter_result["memory"],
                "error": no_filter_result["error"],
            })
            
            # 评估过滤方案
            for threshold in thresholds:
                print(f"  → 过滤(τ={threshold})...", end="", flush=True)
                filter_result = try_evaluate_with_filter(model, images, threshold, device)
                print(f" {'✓' if filter_result['success'] else '✗ OOM'}")
                
                results.append({
                    "sequence_length": seq_len,
                    "method": "filter",
                    "threshold": threshold,
                    "success": filter_result["success"],
                    "time": filter_result["time_total"],
                    "memory": filter_result["memory"],
                    "compression_ratio": filter_result["compression_ratio"],
                    "error": filter_result["error"],
                })
                
                torch.cuda.empty_cache()
                gc.collect()
            
            # 增量保存
            results_df_temp = pd.DataFrame(results)
            results_df_temp.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"\n⚠ 错误 (seq_len={seq_len}): {e}")
            continue
    
    print(f"\n[3/3] 生成图表和报告...")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # 绘制OOM边界
        plot_oom_boundary(
            results_df,
            os.path.join(args.output_dir, "fig_oom_boundary.png")
        )
        
        # 绘制过滤方案的OOM热力图
        plot_oom_heatmap(
            results_df,
            "filter",
            os.path.join(args.output_dir, "fig_oom_heatmap.png")
        )
        
        # 生成统计报告
        report_path = os.path.join(args.output_dir, "eval_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("超长序列评估 —— OOM边界分析报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("测试配置:\n")
            f.write(f"  数据集: {args.dataset_type}\n")
            f.write(f"  序列长度: {sequence_lengths}\n")
            f.write(f"  阈值范围: {thresholds}\n")
            f.write(f"  设备: {device}\n\n")
            
            f.write("OOM统计:\n")
            for method in ["no_filter", "filter"]:
                subset = results_df[results_df["method"] == method]
                success_rate = subset["success"].mean() * 100
                oom_count = (~subset["success"]).sum()
                f.write(f"\n  {method}方案:\n")
                f.write(f"    总测试: {len(subset)}\n")
                f.write(f"    成功: {subset['success'].sum()}\n")
                f.write(f"    OOM: {oom_count}\n")
                f.write(f"    成功率: {success_rate:.1f}%\n")
            
            # 显存统计
            f.write("\n显存统计 (成功情况下的峰值):\n")
            success_df = results_df[results_df["success"] == True]
            if len(success_df) > 0:
                for method in ["no_filter", "filter"]:
                    subset = success_df[success_df["method"] == method]
                    if len(subset) > 0:
                        f.write(f"\n  {method}:\n")
                        f.write(f"    平均显存: {subset['memory'].mean():.0f} MB\n")
                        f.write(f"    最大显存: {subset['memory'].max():.0f} MB\n")
                        f.write(f"    最小显存: {subset['memory'].min():.0f} MB\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("详细结果:\n")
            f.write("=" * 70 + "\n")
            f.write(results_df.to_string(index=False))
        
        print(f"✓ 评估报告: {report_path}")
    
    print(f"\n✅ 评估完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
