import os
import sys
import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT

# --- Global Variables ---
# 实验使用的序列长度列表；控制每次实验输入的视频帧数
SEQUENCE_LENGTHS = [5, 10, 30, 50, 100]  # List of sequence lengths to test
# 要计算相似度的模型块层索引集合；使用模型中的绝对层索引
# 脚本内部会自动只对 aggregator.global_blocks 进行计算，跳过 frame_blocks
BLOCK_LAYERS = list(range(24))  # List of block layers to compute similarity for (0-23)
# 从所有可用 token 中按比例随机采样的比例；用于确定初始采样规模
# 过大可能导致计算开销增大，通常结合 MAX_TOKEN_SAMPLE 共同控制
TOKEN_SAMPLING_PERCENTAGE = 0.1  # Percentage of tokens to randomly sample for similarity calculation
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
    parser = argparse.ArgumentParser("Test Token Similarity vs Block Index vs Sequence Length", add_help=False)
    # General arguments
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")

    parser.add_argument("--dataset_type", choices=["7scenes", "scannet", "images"], 
                       default="7scenes", help="Dataset type to use")
    
    parser.add_argument("--data_dir", type=str, default=PATH_FOR_7SCENES, 
                       help="Path to dataset")
    
    parser.add_argument("--output_dir", type=str, default="./tests/tests_results/token_similarity", 
                       help="Directory to save results")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392],
                       help="Resolution for 7Scenes (H W)")
    
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples per config")
    
    parser.add_argument("--sequence_lengths", type=str, default="5,10,30,50,100", 
                       help="Comma-separated list of sequence lengths to test")
    
    parser.add_argument("--block_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23", 
                       help="Comma-separated list of block layers to compute similarity for")
    
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
    """Set up hooks to extract tokens from specified global block layers only"""
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
    
    # Check if model has aggregator with global_blocks
    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'global_blocks'):
        # Only register hooks for global blocks
        for i, block in enumerate(model.aggregator.global_blocks):
            # Global block indices start from the number of frame blocks
            global_block_idx = i + len(model.aggregator.frame_blocks)
            if global_block_idx in block_layers:
                hook = block.register_forward_hook(hook_fn(f"global_block_{i}"))
                hooks.append(hook)
    
    return tokens, hooks

def calculate_token_similarity(tokens, sampling_percentage=0.1):
    """Calculate average cosine similarity between randomly sampled tokens"""
    if tokens is None or tokens.numel() == 0:
        return 0.0
    
    # Reshape tokens to (num_tokens, feature_dim)
    if len(tokens.shape) == 4:  # (B, H, W, C)
        B, H, W, C = tokens.shape
        num_tokens = B * H * W
        tokens_reshaped = tokens.view(-1, C)
    elif len(tokens.shape) == 3:  # (B, N, C)
        B, N, C = tokens.shape
        num_tokens = B * N
        tokens_reshaped = tokens.view(-1, C)
    else:
        return 0.0
    
    # Determine number of tokens to sample
    num_sample_base = min(max(10, int(num_tokens * sampling_percentage)), num_tokens)
    num_sample = min(num_sample_base, MAX_TOKEN_SAMPLE)
    
    # Randomly sample tokens
    indices = torch.randperm(num_tokens, device=tokens_reshaped.device)[:num_sample]
    sampled_tokens = tokens_reshaped[indices]
    
    # Normalize tokens
    sampled_tokens = torch.nn.functional.normalize(sampled_tokens, dim=1)
    
    if ENABLE_PAIR_SAMPLING and num_sample > PAIRWISE_MATRIX_THRESHOLD:
        K = min(PAIR_SAMPLES, num_sample * PAIR_SAMPLE_MULTIPLIER)
        i = torch.randint(0, num_sample, (K,), device=sampled_tokens.device)
        j = torch.randint(0, num_sample, (K,), device=sampled_tokens.device)
        mask = i != j
        if mask.any():
            sims = (sampled_tokens[i[mask]] * sampled_tokens[j[mask]]).sum(dim=1)
            return sims.mean().item()
        else:
            return 0.0
    else:
        similarity_matrix = torch.matmul(sampled_tokens, sampled_tokens.T)
        upper_tri = similarity_matrix.triu(diagonal=1)
        num_nonzero_pairs = (upper_tri != 0).sum().item()
        if num_nonzero_pairs > 0:
            avg_similarity = upper_tri.sum() / num_nonzero_pairs
            return avg_similarity.item()
        else:
            return 0.0

def generate_heatmap(data, block_layers, sequence_lengths, output_path):
    """Generate and save heatmap of token similarity"""
    # Create pivot table for heatmap
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index="sequence_length", columns="block_layer", values="token_similarity", aggfunc="mean")
    
    # Sort by sequence length and block layer
    pivot = pivot.sort_index(ascending=False)
    
    # Generate heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={"label": "Average Token Similarity"})
    plt.title("Token Similarity vs Block Index vs Sequence Length")
    plt.xlabel("Block Index")
    plt.ylabel("Sequence Length")
    plt.tight_layout()
    
    # Save heatmap
    plt.savefig(output_path)
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
            for block_layer in block_layers:
                # Only process global blocks (skip frame blocks)
                if hasattr(model, 'aggregator'):
                    num_frame_blocks = len(model.aggregator.frame_blocks)
                    if block_layer >= num_frame_blocks:  # Only global blocks
                        global_block_idx = block_layer - num_frame_blocks
                        block_name = f"global_block_{global_block_idx}"
                        
                        if block_name in tokens and tokens[block_name] is not None:
                            similarity = calculate_token_similarity(tokens[block_name], args.token_sampling_percentage)
                            results.append({
                                "dataset": args.dataset_type,
                                "sequence_length": seq_len,
                                "block_layer": block_layer,
                                "token_similarity": similarity
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
        csv_path = os.path.join(args.output_dir, "token_similarity_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")
        
        # Generate and save heatmap
        heatmap_path = os.path.join(args.output_dir, f"token_similarity_heatmap_{args.dataset_type}.png")
        generate_heatmap(df, block_layers, sequence_lengths, heatmap_path)
        print(f"✓ Heatmap saved to {heatmap_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset_type}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Block layers: {block_layers}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
    else:
        print("No results to save")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
