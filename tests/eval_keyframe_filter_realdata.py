"""
关键帧预过滤与特征复用 —— 真实数据评估

基于DINOv2特征的关键帧过滤模块在真实数据集上的全面评估，包括：
  1. 特征区分度: DINOv2 CLS token的余弦距离分布 → 验证可分性
  2. 复用一致性: 直接复用 vs 重新编码的误差分析
  3. 时间开销:   关键帧过滤+特征复用的时间节省
  4. 显存使用:   序列长度vs显存的关系曲线
  5. OOM边界:    不同阈值和序列长度下的Out-of-Memory扫描
  
输出: CSV统计表格 + 中文图表（pdf/png）

使用方式:
  # 使用7Scenes数据集（默认）
  python tests/eval_keyframe_filter_realdata.py \\
    --dataset_type 7scenes \\
    --data_dir /home/hba/Documents/Dataset/7_scenes \\
    --ckpt_path /home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt

  # 使用ScanNet数据集
  python tests/eval_keyframe_filter_realdata.py \\
    --dataset_type scannet \\
    --data_dir /home/hba/Documents/Dataset/ScanNet/scans
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

# 配置中文字体
def setup_chinese_font():
    rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")
    return font_manager.FontProperties(family="SimHei")

FONT_PROP = setup_chinese_font()

# 确保项目根目录在 sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.keyframe_filter import KeyframeFilter

# 数据集默认路径
PATH_FOR_7SCENES = "/home/hba/Documents/Dataset/7_scenes"
PATH_FOR_SCANNET = "/home/hba/Documents/Dataset/ScanNet/scans"


# ============================================================
# 参数解析
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Evaluate Keyframe Filtering on Real Data",
        add_help=False
    )
    # 模型和数据集参数
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="模型检查点路径"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["7scenes", "scannet"],
        default="7scenes",
        help="数据集类型"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PATH_FOR_7SCENES,
        help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tests/eval_keyframe_filter_output",
        help="输出目录"
    )
    # 评估参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[518, 392],
        help="7Scenes 图像分辨率 (H W)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="每个配置测试的样本数"
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="5,10,15,20,30",
        help="序列长度列表（逗号分隔）"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.35,0.5,0.7",
        help="关键帧阈值τ列表（逗号分隔）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    return parser


# ============================================================
# 数据加载函数
# ============================================================

def load_7scenes_data(
    data_dir: str,
    num_frames: int,
    resolution: Tuple[int, int] = (518, 392),
    num_samples: int = 1
) -> torch.Tensor:
    """加载7Scenes数据"""
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
    
    all_images = []
    for idx in range(min(num_samples, len(dataset))):
        views = dataset[idx]
        actual_frames = min(num_frames, len(views))
        selected_views = views[:actual_frames]
        
        imgs = torch.stack([v["img"] for v in selected_views])
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] → [0, 1]
        all_images.append(imgs.unsqueeze(0))
    
    return torch.cat(all_images, dim=0) if all_images else torch.empty(0)


def load_scannet_data(
    data_dir: str,
    num_frames: int,
    num_samples: int = 1
) -> torch.Tensor:
    """加载ScanNet数据"""
    from vggt.utils.eval_utils import (
        get_sorted_image_paths,
        load_images_rgb,
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)])
    
    all_images = []
    for scene in scenes[:num_samples]:
        scene_dir = data_dir / scene
        images_dir = scene_dir / "color"
        if not images_dir.exists():
            continue
        
        image_paths = get_sorted_image_paths(images_dir)
        actual_frames = min(num_frames, len(image_paths))
        selected_paths = image_paths[:actual_frames]
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0) if all_images else torch.empty(0)


# ============================================================
# 核心评估函数
# ============================================================

def evaluate_feature_discrimination(
    model: VGGT,
    images: torch.Tensor,
    device: str = "cuda:0"
) -> Dict[str, np.ndarray]:
    """
    评估DINOv2特征的区分度。
    
    返回:
        dict: {
            "cosine_distances": 相邻帧的余弦距离列表
            "cls_tokens": CLS token张量 [S, D]
        }
    """
    aggregator = model.aggregator
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=0.35)
    
    images = images.to(device, dtype=torch.float32)
    B, S = images.shape[:2]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            cls_tokens, _ = filter_model.extract_features(images)
    
    cls_tokens = cls_tokens[0]  # [S, D]
    cls_norm = F.normalize(cls_tokens.float(), dim=1)
    
    # 计算相邻帧余弦距离
    distances = []
    for i in range(1, S):
        cos_sim = (cls_norm[i] * cls_norm[i-1]).sum().item()
        dist = 1.0 - cos_sim
        distances.append(dist)
    
    return {
        "cosine_distances": np.array(distances),
        "cls_tokens": cls_tokens.cpu().numpy()
    }


def evaluate_reuse_consistency(
    model: VGGT,
    images: torch.Tensor,
    threshold: float = 0.35,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    评估特征复用的一致性。
    
    对比：直接计算深层特征 vs 使用预计算patch token后计算
    """
    aggregator = model.aggregator
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=threshold)
    
    images = images.to(device, dtype=torch.float32)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # 路径1: 直接输入（无过滤、无复用）
            pred1 = model(images)
            depth1 = pred1["depth"] if "depth" in pred1 else None
            
            # 路径2: 过滤后使用复用特征
            filter_result = filter_model(images)
            filtered_images = filter_result["filtered_images"].to(device, dtype=torch.float32)
            patch_tokens = filter_result["patch_tokens"].to(device, dtype=torch.bfloat16)
            
            pred2 = model(filtered_images, precomputed_patch_tokens=patch_tokens)
            depth2 = pred2["depth"] if "depth" in pred2 else None
    
    # 计算深度误差（仅在两个预测都存在时）
    errors = {}
    if depth1 is not None and depth2 is not None:
        # depth1: [B, S, H, W, 1], depth2: [B, K, H, W, 1]
        B, K = depth2.shape[:2]
        # 只对关键帧位置进行比较
        keyframe_indices = filter_result["keyframe_indices"][0]
        depth1_selected = depth1[0, keyframe_indices]  # [K, H, W, 1]
        
        # 计算L2误差
        mse = torch.mean((depth1_selected - depth2[0]) ** 2).item()
        errors["depth_mse"] = mse
    
    return errors


def evaluate_timing(
    model: VGGT,
    images: torch.Tensor,
    threshold: float = 0.35,
    num_iterations: int = 5,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    评估时间开销。
    
    returns:
        dict: {
            "time_no_filter": 无过滤的推理时间
            "time_filter_only": 仅关键帧过滤时间
            "time_filtered_reuse": 过滤+复用的推理时间
            "speedup": 加速比
        }
    """
    aggregator = model.aggregator
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=threshold)
    
    images = images.to(device, dtype=torch.float32)
    
    # 预热
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(images)
            torch.cuda.synchronize(device)
    
    # 测试1: 无过滤
    times_no_filter = []
    for _ in range(num_iterations):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(images)
        torch.cuda.synchronize(device)
        times_no_filter.append(time.time() - t0)
    
    time_no_filter = np.mean(times_no_filter[1:])  # 排除第一次
    
    # 测试2: 过滤 + 复用
    times_filter = []
    times_inference = []
    for _ in range(num_iterations):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
        # 过滤时间
        t0 = time.time()
        filter_result = filter_model(images)
        torch.cuda.synchronize(device)
        t_filter = time.time() - t0
        
        filtered_images = filter_result["filtered_images"].to(device, dtype=torch.float32)
        patch_tokens = filter_result["patch_tokens"].to(device, dtype=torch.bfloat16)
        
        # 推理时间
        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(filtered_images, precomputed_patch_tokens=patch_tokens)
        torch.cuda.synchronize(device)
        t_inference = time.time() - t0
        
        times_filter.append(t_filter)
        times_inference.append(t_inference)
    
    time_filter = np.mean(times_filter[1:])
    time_inference = np.mean(times_inference[1:])
    time_total = time_filter + time_inference
    
    # 压缩率
    compression_ratio = filter_result["stats"]["compression_ratio"]
    
    return {
        "time_no_filter": time_no_filter,
        "time_filter": time_filter,
        "time_inference_filtered": time_inference,
        "time_total": time_total,
        "compression_ratio": compression_ratio,
        "speedup": time_no_filter / time_total if time_total > 0 else 1.0,
    }


def evaluate_memory_usage(
    model: VGGT,
    images: torch.Tensor,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    评估显存使用。
    """
    images = images.to(device, dtype=torch.float32)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(images)
    
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
    
    return {
        "peak_memory_mb": peak_memory,
        "per_frame_memory_mb": peak_memory / images.shape[1]
    }


# ============================================================
# 绘图函数
# ============================================================

def plot_cosine_distance_distribution(
    distances_by_seq: Dict[int, np.ndarray],
    output_path: str
):
    """绘制余弦距离分布直方图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (seq_len, distances) in enumerate(sorted(distances_by_seq.items())):
        if idx >= 6:
            break
        ax = axes[idx]
        ax.hist(distances, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        ax.set_title(f"序列长度 {seq_len}", fontsize=12, fontproperties=FONT_PROP)
        ax.set_xlabel("余弦距离", fontproperties=FONT_PROP)
        ax.set_ylabel("频数", fontproperties=FONT_PROP)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存余弦距离分布图: {output_path}")


def plot_threshold_vs_retention(
    results_df: pd.DataFrame,
    output_path: str
):
    """绘制阈值vs保留帧率曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for seq_len in sorted(results_df["sequence_length"].unique()):
        subset = results_df[results_df["sequence_length"] == seq_len]
        subset = subset.sort_values("threshold")
        ax.plot(
            subset["threshold"],
            subset["compression_ratio"] * 100,
            marker="o",
            label=f"序列长度 {seq_len}",
            linewidth=2
        )
    
    ax.set_xlabel("阈值 τ", fontsize=12, fontproperties=FONT_PROP)
    ax.set_ylabel("保留帧率 (%)", fontsize=12, fontproperties=FONT_PROP)
    ax.set_title("关键帧阈值与保留比例", fontsize=14, fontproperties=FONT_PROP)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存阈值vs保留帧率图: {output_path}")


def plot_timing_speedup(
    results_df: pd.DataFrame,
    output_path: str
):
    """绘制时间节省和加速比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 时间对比
    seq_lens = sorted(results_df["sequence_length"].unique())
    time_no_filter = results_df.groupby("sequence_length")["time_no_filter"].mean()
    time_total = results_df.groupby("sequence_length")["time_total"].mean()
    
    x = np.arange(len(seq_lens))
    width = 0.35
    ax1.bar(x - width/2, time_no_filter.values, width, label="无过滤", alpha=0.8)
    ax1.bar(x + width/2, time_total.values, width, label="过滤+复用", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lens)
    ax1.set_xlabel("序列长度", fontsize=11, fontproperties=FONT_PROP)
    ax1.set_ylabel("推理时间 (s)", fontsize=11, fontproperties=FONT_PROP)
    ax1.set_title("推理时间对比", fontsize=12, fontproperties=FONT_PROP)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # 加速比
    speedup = results_df.groupby("sequence_length")["speedup"].mean()
    ax2.plot(seq_lens, speedup.values, marker="s", linewidth=2, markersize=8, color="green")
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="无加速")
    ax2.set_xlabel("序列长度", fontsize=11, fontproperties=FONT_PROP)
    ax2.set_ylabel("加速比", fontsize=11, fontproperties=FONT_PROP)
    ax2.set_title("端到端加速倍数", fontsize=12, fontproperties=FONT_PROP)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存时间加速图: {output_path}")


def plot_memory_vs_sequence(
    results_df: pd.DataFrame,
    output_path: str
):
    """绘制显存使用vs序列长度"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memory_data = results_df.groupby("sequence_length")["peak_memory_mb"].mean()
    ax.plot(
        memory_data.index,
        memory_data.values,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="purple"
    )
    ax.fill_between(
        memory_data.index,
        memory_data.values,
        alpha=0.3,
        color="purple"
    )
    ax.set_xlabel("序列长度", fontsize=12, fontproperties=FONT_PROP)
    ax.set_ylabel("显存 (MB)", fontsize=12, fontproperties=FONT_PROP)
    ax.set_title("显存使用与序列长度", fontsize=14, fontproperties=FONT_PROP)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存显存使用图: {output_path}")


# ============================================================
# 主程序
# ============================================================

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print("关键帧预过滤与特征复用 —— 真实数据评估")
    print("=" * 70)
    
    # 加载模型
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] 加载模型...")
    
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
    else:
        print(f"⚠ 未找到检查点: {args.ckpt_path}，使用随机初始化")
    
    model = model.to(device).eval()
    # 注意：不转为float16，使用torch.cuda.amp.autocast处理混合精度
    
    # 解析参数
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    
    # 收集结果
    results = []
    distances_by_seq = {}
    
    print(f"\n[2/5] 数据加载与特征区分度评估...")
    
    for seq_len in tqdm(sequence_lengths, desc="序列长度"):
        try:
            if args.dataset_type == "7scenes":
                images = load_7scenes_data(
                    args.data_dir,
                    seq_len,
                    tuple(args.resolution),
                    args.num_samples
                )
            else:  # scannet
                images = load_scannet_data(
                    args.data_dir,
                    seq_len,
                    args.num_samples
                )
            
            if images.shape[0] == 0:
                print(f"⚠ 序列长度 {seq_len}: 无法加载数据")
                continue
            
            images = images[:1]  # 仅用第一个样本
            
            # 评估特征区分度
            disc_result = evaluate_feature_discrimination(model, images, device)
            distances_by_seq[seq_len] = disc_result["cosine_distances"]
            
            # 评估不同阈值下的性能
            for threshold in thresholds:
                try:
                    timing_result = evaluate_timing(
                        model,
                        images,
                        threshold=threshold,
                        device=device
                    )
                    
                    memory_result = evaluate_memory_usage(model, images, device)
                    
                    results.append({
                        "sequence_length": seq_len,
                        "threshold": threshold,
                        "compression_ratio": timing_result["compression_ratio"],
                        "time_no_filter": timing_result["time_no_filter"],
                        "time_filter": timing_result["time_filter"],
                        "time_inference_filtered": timing_result["time_inference_filtered"],
                        "time_total": timing_result["time_total"],
                        "speedup": timing_result["speedup"],
                        "peak_memory_mb": memory_result["peak_memory_mb"],
                    })
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠ OOM: 序列长度={seq_len}, 阈值={threshold}")
                    torch.cuda.empty_cache()
                    continue
                
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"⚠ 错误 (seq_len={seq_len}): {e}")
            continue
    
    print(f"\n[3/5] 生成中文图表...")
    
    # 绘制余弦距离分布
    if distances_by_seq:
        plot_cosine_distance_distribution(
            distances_by_seq,
            os.path.join(args.output_dir, "fig_cosine_distance_dist.png")
        )
    
    # 绘制阈值vs保留帧率
    if results:
        results_df = pd.DataFrame(results)
        
        plot_threshold_vs_retention(
            results_df,
            os.path.join(args.output_dir, "fig_threshold_vs_retention.png")
        )
        
        plot_timing_speedup(
            results_df,
            os.path.join(args.output_dir, "fig_timing_speedup.png")
        )
        
        plot_memory_vs_sequence(
            results_df,
            os.path.join(args.output_dir, "fig_memory_vs_sequence.png")
        )
    
    print(f"\n[4/5] 保存CSV结果表...")
    
    # 保存详细结果
    if results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, "eval_keyframe_filter_detailed.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"✓ 详细结果: {csv_path}")
        
        # 保存汇总表
        summary_df = results_df.groupby("sequence_length").agg({
            "compression_ratio": "mean",
            "speedup": "mean",
            "peak_memory_mb": "mean",
        }).round(4)
        
        summary_path = os.path.join(args.output_dir, "eval_keyframe_filter_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"✓ 汇总表: {summary_path}")
    
    print(f"\n[5/5] 生成评估报告...")
    
    # 打印统计信息
    if results:
        print("\n" + "=" * 70)
        print("评估结果汇总")
        print("=" * 70)
        print(f"数据集: {args.dataset_type}")
        print(f"样本数: {args.num_samples}")
        print(f"序列长度: {sequence_lengths}")
        print(f"阈值范围: {min(thresholds):.2f} - {max(thresholds):.2f}")
        print("\n主要指标:")
        print(f"  平均压缩率: {results_df['compression_ratio'].mean():.1%}")
        print(f"  平均加速比: {results_df['speedup'].mean():.2f}×")
        print(f"  峰值显存: {results_df['peak_memory_mb'].max():.0f} MB")
        print(f"  最长序列: {results_df['sequence_length'].max()} 帧")
        print("=" * 70)
        
        # 生成文本报告
        report_path = os.path.join(args.output_dir, "eval_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("关键帧预过滤与特征复用评估报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"数据集: {args.dataset_type}\n")
            f.write(f"推理设备: {device}\n")
            f.write(f"模型检查点: {args.ckpt_path}\n\n")
            f.write("评估参数:\n")
            f.write(f"  样本数: {args.num_samples}\n")
            f.write(f"  序列长度: {sequence_lengths}\n")
            f.write(f"  阈值范围: {min(thresholds):.2f} - {max(thresholds):.2f}\n\n")
            f.write("主要结果:\n")
            f.write(f"  平均压缩率: {results_df['compression_ratio'].mean():.1%}\n")
            f.write(f"  平均加速比: {results_df['speedup'].mean():.2f}×\n")
            f.write(f"  峰值显存: {results_df['peak_memory_mb'].max():.0f} MB\n")
            f.write(f"  最长序列: {results_df['sequence_length'].max()} 帧\n\n")
            f.write("详细结果表:\n\n")
            f.write(results_df.to_string(index=False))
        
        print(f"✓ 评估报告: {report_path}")
    
    print(f"\n✅ 评估完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
