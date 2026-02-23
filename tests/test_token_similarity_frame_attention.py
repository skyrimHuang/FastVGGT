import os
import sys
import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

def _setup_chinese_font():
    """Ensure SimHei font is registered and used for Chinese text rendering."""
    mpl_data = matplotlib.get_data_path()
    font_candidates = [
        os.path.join(mpl_data, "fonts", "ttf", "SimHei.ttf"),
        os.path.join(mpl_data, "fonts", "ttf", "simhei.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "simhei.ttf"),
    ]

    font_path = None
    for candidate in font_candidates:
        if os.path.exists(candidate):
            font_path = candidate
            break

    if font_path:
        font_manager.fontManager.addfont(font_path)
        if hasattr(font_manager, "_load_fontmanager"):
            font_manager._load_fontmanager(try_read_cache=False)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_name = "SimHei"

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name, "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题
    plt.rcParams["figure.constrained_layout.use"] = False
    sns.set_style("whitegrid", {"font.sans-serif": plt.rcParams["font.sans-serif"]})

    return font_manager.FontProperties(fname=font_path) if font_path else font_manager.FontProperties(family="SimHei")


FONT_PROP = _setup_chinese_font()

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT

# --- Global Variables ---
# 模型 VGGT 的合并比例参数；设为 0.0 表示不合并， > 0 表示合并该比例的 token
MERGE_RATIO = 0.0  # Fixed merge ratio for all experiments (0.0 = no merging, >0 = merge ratio)
# 参与相似度计算的 token 数量上限（最终采样规模的硬限制），用于防止 s×s 矩阵或配对采样规模过大
MAX_TOKEN_SAMPLE = 2048
# 当采样的 token 数量超过该阈值时，不再构建完整的 s×s 相似度矩阵，切换为配对采样近似
# 该值用于控制是否走近似路径，避免 O(n^2) 的内存与计算
PAIRWISE_MATRIX_THRESHOLD = 1024
# 配对采样的最大样本对数上限 K；数值越大近似越稳定，但计算时间也越长
# 与特征维度 d 成线性关系，默认 250000 在常见设置下较为稳妥
PAIR_SAMPLES = 250_000
# 配对采样的规模随采样 token 数量的倍率上限；实际 K = min(PAIR_SAMPLES, num_sample * PAIR_SAMPLE_MULTIPLIER)
# 当采样 token 很少时避免过度采样，当很多时避免超过上限
PAIR_SAMPLE_MULTIPLIER = 10
# 是否启用配对采样近似；关闭则始终构建完整相似度矩阵（可能非常慢/占用显存）
ENABLE_PAIR_SAMPLING = True

# Default paths
PATH_FOR_7SCENES = "/home/hba/Documents/Dataset/7_scenes"
PATH_FOR_SCANNET = "/home/hba/Documents/Dataset/ScanNet/scans"

# --- Helper Functions ---

def get_args_parser():
    parser = argparse.ArgumentParser("Test Frame Attention Token Similarity vs Block Index vs Sequence Length", add_help=False)
    # General arguments
    parser.add_argument("--ckpt_path", type=str, default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt", help="Path to the model checkpoint")

    parser.add_argument("--dataset_type", choices=["7scenes", "scannet", "images"], 
                       default="7scenes", help="Dataset type to use")
    
    parser.add_argument("--data_dir", type=str, default=PATH_FOR_7SCENES, 
                       help="Path to dataset")
    
    parser.add_argument("--output_dir", type=str, default="./tests/tests_result/token_similarity", 
                       help="Directory to save results")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392],
                       help="Resolution for 7Scenes (H W)")
    
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples per config")
    
    parser.add_argument("--sequence_lengths", type=str, default="5,10", 
                       help="Comma-separated list of sequence lengths to test")
    
    parser.add_argument("--block_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23", 
                       help="Comma-separated list of frame block layers to compute similarity for")
    
    parser.add_argument("--token_sampling_percentage", type=float, default=0.1, 
                       help="Percentage of tokens to randomly sample for similarity calculation")
    
    parser.add_argument("--vis_attn_map", action="store_true", 
                       help="Visualize attention maps during inference")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser


# --- Data Loading Functions ---

def load_7scenes_data(
    data_dir: str,
    num_frames: int,
    resolution: Tuple[int, int] = (518, 392),
    num_samples: int = 1
) -> tuple:
    """Load 7Scenes data.
    
    Returns:
        tuple: (images, image_paths) where images is [num_samples, num_frames, C, H, W] 
               and image_paths is list of image file paths for visualization
    """
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
    all_image_paths = []
    for idx in range(min(num_samples, len(dataset))):
        views = dataset[idx]
        actual_frames = min(num_frames, len(views))
        selected_views = views[:actual_frames]
        
        # Normalize and stack
        imgs = torch.stack([v["img"] for v in selected_views])
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        all_images.append(imgs.unsqueeze(0))
        
        # Collect image paths if available
        for view in selected_views:
            if "img_path" in view:
                all_image_paths.append(view["img_path"])
    
    return torch.cat(all_images, dim=0), all_image_paths


def load_scannet_data(
    data_dir: str,
    num_frames: int,
    num_samples: int = 1
) -> tuple:
    """Load ScanNet data.
    
    Returns:
        tuple: (images, image_paths) where images is tensor and image_paths is list
    """
    from vggt.utils.eval_utils import (
        get_sorted_image_paths, 
        load_images_rgb, 
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)])
    
    all_images = []
    all_image_paths = []
    for scene in scenes[:num_samples]:
        scene_dir = data_dir / scene
        images_dir = scene_dir / "color"
        image_paths = get_sorted_image_paths(images_dir)
        
        actual_frames = min(num_frames, len(image_paths))
        selected_paths = image_paths[:actual_frames]
        all_image_paths.extend(selected_paths)
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0), all_image_paths


def load_images(
    data_dir: str,
    num_frames: int,
    num_samples: int = 1
) -> tuple:
    """Load generic images.
    
    Returns:
        tuple: (images, image_paths) where images is tensor and image_paths is list
    """
    from vggt.utils.eval_utils import (
        get_sorted_image_paths,
        load_images_rgb,
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    image_paths = get_sorted_image_paths(data_dir)
    
    all_images = []
    all_image_paths = []
    total_images = len(image_paths)
    stride = max(1, total_images // num_samples)
    
    for idx in range(num_samples):
        start_idx = idx * stride
        if start_idx >= total_images:
            break
        end_idx = min(start_idx + num_frames, total_images)
        selected_paths = image_paths[start_idx:end_idx]
        all_image_paths.extend(selected_paths)
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0), all_image_paths

def setup_model_hooks(model, block_layers):
    """Set up hooks to extract tokens from specified frame block layers only.
    
    Note: block_layers indices refer to frame_block indices (0-12 for frame blocks),
    NOT the global_block indices.
    """
    tokens = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # Extract tokens from the output
            if isinstance(output, tuple):
                # For blocks that return multiple values
                tokens[name] = output[0].detach()
            else:
                # For blocks that return a single value
                tokens[name] = output.detach()
        return hook
    
    # Register hooks for specified block layers
    hooks = []
    
    # Check if model has aggregator with frame_blocks
    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'frame_blocks'):
        # Register hooks for frame blocks using their local indices
        for i, block in enumerate(model.aggregator.frame_blocks):
            # Compare local frame_block index (i) with block_layers
            if i in block_layers:
                hook = block.register_forward_hook(hook_fn(f"frame_block_{i}"))
                hooks.append(hook)
    else:
        print("Warning: model does not have frame_blocks. Skipping hook setup.")
    
    return tokens, hooks

def calculate_token_similarity(tokens, seq_len, sampling_percentage=0.1, num_repeats=5, debug=False):
    """计算 frame block 中帧内 token 的相似度（不计算帧间相似度）。
    
    关键特征：
    - 将 tokens 按帧分割
    - 对每一帧内部的 token 计算相似度
    - 对所有帧的相似度求平均
    
    Args:
        tokens: 模型 hook 输出的 token 张量，形状 [B*T, H, W, D] 或 [B*T, N, D]
        seq_len: 序列长度 (帧数)
        sampling_percentage: 采样百分比 (0.0-1.0)
        num_repeats: 独立采样运行次数（默认：5）
        debug: 是否打印调试信息
    
    Returns:
        float: 所有帧平均的 token 相似度
    """
    if tokens is None or tokens.numel() == 0:
        return 0.0
    
    # 获取 token 维度信息
    if len(tokens.shape) == 4:  # (B*T, H, W, C)
        B_times_T, H, W, C = tokens.shape
        tokens_flat = tokens.view(-1, C)
        tokens_per_frame = H * W
    elif len(tokens.shape) == 3:  # (B*T, N, C)
        B_times_T, N, C = tokens.shape
        tokens_flat = tokens.view(-1, C)
        tokens_per_frame = N
    else:
        return 0.0
    
    total_tokens = tokens_flat.shape[0]
    feature_dim = tokens_flat.shape[1]
    
    # 计算每帧的 token 数
    tokens_per_frame_actual = total_tokens // seq_len
    remainder = total_tokens % seq_len
    
    if debug:
        print(f"  [DEBUG] Token shape: {tokens.shape} -> flattened: {tokens_flat.shape}")
        print(f"  [DEBUG] Sequence length: {seq_len}, Tokens per frame: {tokens_per_frame_actual} (remainder: {remainder})")
        print(f"  [DEBUG] Feature dimension: {feature_dim}")
    
    # 按帧分割并计算每帧的相似度
    frame_similarities = []
    
    for frame_idx in range(seq_len):
        # 提取当前帧的 tokens
        frame_start = frame_idx * tokens_per_frame_actual
        frame_end = frame_start + tokens_per_frame_actual
        frame_tokens = tokens_flat[frame_start:frame_end]  # [tokens_per_frame, D]
        
        if frame_tokens.shape[0] == 0:
            continue
        
        # 决定采样数量
        num_frame_tokens = frame_tokens.shape[0]
        num_sample_base = min(max(10, int(num_frame_tokens * sampling_percentage)), num_frame_tokens)
        num_sample = min(num_sample_base, MAX_TOKEN_SAMPLE)
        
        if debug and frame_idx == 0:
            print(f"  [DEBUG] Frame 0: {num_frame_tokens} tokens, sampling {num_sample} ({sampling_percentage*100:.1f}%)")
        
        # 执行多次独立采样
        frame_repeat_similarities = []
        
        for repeat_idx in range(num_repeats):
            # 随机采样 token
            indices = torch.randperm(num_frame_tokens, device=frame_tokens.device)[:num_sample]
            sampled_tokens = frame_tokens[indices]
            
            # 归一化 token 用于余弦相似度
            sampled_tokens = torch.nn.functional.normalize(sampled_tokens, dim=1)
            
            if ENABLE_PAIR_SAMPLING and num_sample > PAIRWISE_MATRIX_THRESHOLD:
                # 配对采样
                K = min(PAIR_SAMPLES, num_sample * PAIR_SAMPLE_MULTIPLIER)
                i = torch.randint(0, num_sample, (K,), device=sampled_tokens.device)
                j = (i + torch.randint(1, num_sample, (K,), device=sampled_tokens.device)) % num_sample
                sims = (sampled_tokens[i] * sampled_tokens[j]).sum(dim=1)
                avg_sim = sims.mean().item()
            else:
                # 完整相似度矩阵
                similarity_matrix = torch.matmul(sampled_tokens, sampled_tokens.T)
                upper_tri = similarity_matrix.triu(diagonal=1)
                num_pairs = num_sample * (num_sample - 1) // 2
                avg_sim = upper_tri.sum().item() / num_pairs if num_pairs > 0 else 0.0
            
            frame_repeat_similarities.append(avg_sim)
        
        # 该帧的平均相似度（多次采样的平均）
        frame_avg_sim = np.mean(frame_repeat_similarities)
        frame_similarities.append(frame_avg_sim)
        
        if debug and frame_idx == 0:
            print(f"  [DEBUG] Frame 0 similarity (across {num_repeats} repeats): mean={frame_avg_sim:.4f}, std={np.std(frame_repeat_similarities):.4f}")
    
    # 所有帧相似度的平均值
    if len(frame_similarities) == 0:
        return 0.0
    
    final_avg = np.mean(frame_similarities)
    
    if debug:
        print(f"  [DEBUG] Across {seq_len} frames: mean={final_avg:.4f}, std={np.std(frame_similarities):.4f}")
    
    return final_avg

def generate_heatmap(data, block_layers, sequence_lengths, output_path):
    """生成并保存 frame attention token 相似度热力图"""
    # 创建透视表用于热力图
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index="sequence_length", columns="block_layer", values="token_similarity", aggfunc="mean")
    
    # 按序列长度和块层排序
    pivot = pivot.sort_index(ascending=False)
    
    # 计算合适的图像尺寸：根据块数量动态调整宽度
    num_blocks = len(block_layers)
    num_seq_lengths = len(sequence_lengths)
    # 每个块至少0.8英寸宽，确保数字不重叠
    fig_width = max(15, num_blocks * 0.8)
    fig_height = max(10, num_seq_lengths * 0.6)
    
    # 生成热力图
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        cbar_kws={"label": "平均 Token 相似度"},
        annot_kws={"fontsize": 10, "fontproperties": FONT_PROP},
        ax=ax,
    )
    ax.set_xlabel("帧注意力块索引", fontsize=12, fontproperties=FONT_PROP)
    ax.set_ylabel("序列长度", fontsize=12, fontproperties=FONT_PROP)
    # ax.set_title("帧注意力 Token 相似度 vs 块索引 vs 序列长度", fontsize=14, fontproperties=FONT_PROP)

    # 应用中文字体到刻度与颜色条
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=FONT_PROP)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=FONT_PROP)
    if ax.collections and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.set_label("平均 Token 相似度", fontproperties=FONT_PROP)
    
    # 保存热力图
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def main(args):
    """Main function to run the experiment"""
    # --- Setup for Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # --- Parse Sequence Lengths and Block Layers ---
    sequence_lengths = [int(s) for s in args.sequence_lengths.split(",")]
    block_layers = [int(b) for b in args.block_layers.split(",")]
    
    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Initialize Model ---
    # When MERGE_RATIO = 0.0, disable merging by setting merging to 25 (blocks only go up to 23)
    # When MERGE_RATIO > 0, enable merging from block 0
    merging_threshold = 25 if MERGE_RATIO == 0.0 else 0
    model = VGGT(merging=merging_threshold, merge_ratio=MERGE_RATIO, enable_point=True, 
                 enable_depth=True, enable_camera=True, vis_attn_map=args.vis_attn_map)
    
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {args.ckpt_path}")
    else:
        print(f"Checkpoint file not found at: {args.ckpt_path}")
        return
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = model.to(device).eval().to(torch.float16)
    
    # --- Setup Model Hooks ---
    tokens, hooks = setup_model_hooks(model, block_layers)
    
    # --- Experiment Results ---
    results = []
    
    # --- Experiment Loop ---
    for seq_len in tqdm(sequence_lengths, desc="Testing"):
        # Load data
        try:
            if args.dataset_type == "7scenes":
                images, image_paths = load_7scenes_data(args.data_dir, seq_len, 
                                          tuple(args.resolution), args.num_samples)
            elif args.dataset_type == "scannet":
                images, image_paths = load_scannet_data(args.data_dir, seq_len, args.num_samples)
            else:  # images
                images, image_paths = load_images(args.data_dir, seq_len, args.num_samples)
        except Exception as e:
            print(f"Error loading data for seq_len={seq_len}: {e}")
            continue
        
        # Forward pass to trigger hooks
        images_device = images.to(device, dtype=torch.float16)
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Pass image_paths if visualization is enabled
                    if args.vis_attn_map and image_paths:
                        model(images_device, image_paths=image_paths)
                    else:
                        model(images_device)
            
            # Calculate token similarity for each block layer
            # FIX: Calculate multiple times per block and average to reduce variance
            for block_layer in block_layers:
                # block_layer is a frame_block index (0-23)
                block_name = f"frame_block_{block_layer}"
                
                if block_name in tokens and tokens[block_name] is not None:
                    # Calculate similarity 3 times per block and average
                    similarities_per_block = []
                    for calc_idx in range(3):
                        # Enable debug only on first calculation of first sequence length
                        debug_enabled = (seq_len == sequence_lengths[0] and block_layer == block_layers[0] and calc_idx == 0)
                        similarity = calculate_token_similarity(
                            tokens[block_name],
                            seq_len=seq_len,
                            sampling_percentage=args.token_sampling_percentage,
                            num_repeats=5,
                            debug=debug_enabled
                        )
                        similarities_per_block.append(similarity)
                    
                    # Average across 3 calculations
                    avg_similarity = np.mean(similarities_per_block)
                    results.append({
                        "sequence_length": seq_len,
                        "block_layer": block_layer,
                        "token_similarity": avg_similarity
                    })
            
            # Clear tokens for next iteration
            for key in tokens:
                tokens[key] = None
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at sequence_length={seq_len}")
        
        # Clear GPU memory and run garbage collection
        del images_device, images
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- Remove Hooks ---
    for hook in hooks:
        hook.remove()
    
    # --- Save Results ---
    if results:
        # Save to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, "token_similarity_frame_attention_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")
        
        # Generate and save heatmap
        heatmap_path = os.path.join(args.output_dir, f"token_similarity_frame_attention_{args.dataset_type}.png")
        generate_heatmap(df, block_layers, sequence_lengths, heatmap_path)
        print(f"✓ Heatmap saved to {heatmap_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset_type}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Block layers (Frame Attention): {block_layers}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
    else:
        print("No results to save")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
