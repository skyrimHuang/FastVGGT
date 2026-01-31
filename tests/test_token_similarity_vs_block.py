import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Global Variables ---
SEQUENCE_LENGTHS = [5, 10, 30, 50, 100]  # List of sequence lengths to test
BLOCK_LAYERS = list(range(24))  # List of block layers to compute similarity for (0-23)
TOKEN_SAMPLING_PERCENTAGE = 0.1  # Percentage of tokens to randomly sample for similarity calculation
MERGE_RATIO = 0.9  # Fixed merge ratio for all experiments

# --- Helper Functions ---
def get_args_parser():
    parser = argparse.ArgumentParser("Test Token Similarity vs Block Index vs Sequence Length", add_help=False)
    # General arguments
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="e:/GraduateProject/FastVGGT/tests/results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--size", type=int, default=518, help="Image size for the model")
    parser.add_argument("--kf", type=int, default=2, help="Keyframe interval")
    parser.add_argument("--use_proj", action="store_true", help="Use Umeyama alignment instead of only ICP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Sequence length arguments
    parser.add_argument("--sequence_lengths", type=str, default="5,10,30,50,100", help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--block_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23", help="Comma-separated list of block layers to compute similarity for")
    parser.add_argument("--token_sampling_percentage", type=float, default=0.1, help="Percentage of tokens to randomly sample for similarity calculation")
    
    # 7-Scenes dataset arguments
    parser.add_argument("--7scenes_data_root", type=str, default=None, help="Path to the 7-Scenes dataset root")
    
    # ScanNet dataset arguments
    parser.add_argument("--scannet_data_dir", type=Path, default=None, help="Path to the ScanNet dataset root")
    parser.add_argument("--scannet_gt_ply_dir", type=Path, default=None, help="Path to the ScanNet GT ply files")
    parser.add_argument("--scannet_input_frame", type=int, default=100, help="Maximum number of frames selected for processing per ScanNet scene")
    parser.add_argument("--scannet_depth_conf_thresh", type=float, default=1.0, help="Depth confidence threshold for ScanNet")
    parser.add_argument("--scannet_chamfer_max_dist", type=float, default=0.5, help="Maximum distance threshold in Chamfer Distance computation for ScanNet")
    
    return parser

def load_7scenes_data(args, seq_len):
    """Load 7-Scenes dataset for a given sequence length"""
    from data import SevenScenes
    
    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError(f"Resolution for size {args.size} not implemented.")
    
    dataset = SevenScenes(
        ROOT=args.sevenscenes_data_root,
        num_frames=seq_len,  # 使用num_frames而不是num_seq
        full_video=True,
        kf_every=args.kf,
        split="test",
        resolution=resolution,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=default_collate)
    return dataloader

def load_scannet_data(args, seq_len):
    """Load ScanNet dataset for a given sequence length"""
    from vggt.utils.eval_utils import (
        load_poses,
        get_vgg_input_imgs,
        get_sorted_image_paths,
        build_frame_selection,
        load_images_rgb,
    )
    
    # Get all scenes
    scene_dirs = [d for d in args.scannet_data_dir.iterdir() if d.is_dir()]
    if not scene_dirs:
        return None
    
    # Use first scene for testing
    scene_dir = scene_dirs[0]
    images_dir = scene_dir / "color"
    pose_path = scene_dir / "pose"
    
    # Get image paths and poses
    image_paths = get_sorted_image_paths(images_dir)
    poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
    
    if poses_gt is None or first_gt_pose is None or available_pose_frame_ids is None:
        return None
    
    # Frame filtering - use sequence length as input frame limit
    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
        image_paths, available_pose_frame_ids, seq_len
    )
    
    if len(selected_image_paths) == 0:
        return None
    
    # Load images
    images = load_images_rgb(selected_image_paths)
    if not images or len(images) < 3:
        return None
    
    # Prepare VGGT input
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    
    return {
        "vgg_input": vgg_input,
        "patch_width": patch_width,
        "patch_height": patch_height,
        "selected_frame_ids": selected_frame_ids,
        "selected_image_paths": selected_image_paths,
        "selected_pose_indices": selected_pose_indices,
        "poses_gt": poses_gt,
        "first_gt_pose": first_gt_pose,
    }

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
    num_sample = min(max(10, int(num_tokens * sampling_percentage)), num_tokens)
    
    # Randomly sample tokens
    indices = torch.randperm(num_tokens, device=tokens_reshaped.device)[:num_sample]
    sampled_tokens = tokens_reshaped[indices]
    
    # Normalize tokens
    sampled_tokens = torch.nn.functional.normalize(sampled_tokens, dim=1)
    
    # Calculate pairwise cosine similarity
    similarity_matrix = torch.matmul(sampled_tokens, sampled_tokens.T)
    
    # Extract upper triangle (excluding diagonal)
    upper_tri = similarity_matrix.triu(diagonal=1)
    
    # Calculate average similarity (only non-zero pairs)
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
    from vggt.models.vggt import VGGT
    
    model = VGGT(merging=0, merge_ratio=MERGE_RATIO, enable_point=True)
    try:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    except FileNotFoundError:
        print(f"Checkpoint file not found at: {args.ckpt_path}")
        return
    
    # --- Determine Data Type ---
    dtype = torch.bfloat16
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print("WARNING: bfloat16 not supported on this GPU, falling back to float16")
        dtype = torch.float16
    
    model = model.to(args.device).eval().to(dtype)
    
    # --- Setup Model Hooks ---
    tokens, hooks = setup_model_hooks(model, block_layers)
    
    # --- Experiment Results ---
    results = []
    
    # --- Experiment Loop ---
    for seq_len in sequence_lengths:
        print(f"Running experiment with Sequence Length: {seq_len}")
        
        # --- Process 7-Scenes Dataset if Available ---
        if args.sevenscenes_data_root is not None:
            try:
                dataloader = load_7scenes_data(args, seq_len)
                
                # 只处理前3个批次以避免过度计算
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing 7-Scenes with SeqLen {seq_len}")):
                    if batch_idx >= 3:  # 限制处理的批次数量
                        break
                    # Move data to device
                    for view in batch:
                        for name, tensor in view.items():
                            if isinstance(tensor, torch.Tensor):
                                view[name] = tensor.to(args.device, non_blocking=True)
                            elif isinstance(tensor, (list, tuple)):
                                view[name] = [x.to(args.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in tensor]
                    
                    # Normalize images
                    for view in batch:
                        view["img"] = (view["img"] + 1.0) / 2.0
                    imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)
                    
                    # Forward pass to trigger hooks
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                        model(imgs_tensor)
                    
                    # Calculate token similarity for each global block layer
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
                                        "dataset": "7-Scenes",
                                        "sequence_length": seq_len,
                                        "block_layer": block_layer,
                                        "token_similarity": similarity
                                    })
                    
                    # Clear tokens for next iteration
                    for key in tokens:
                        tokens[key] = None
                    
            except Exception as e:
                print(f"Error processing 7-Scenes dataset: {e}")
        
        # --- Process ScanNet Dataset if Available ---
        if args.scannet_data_dir is not None:
            try:
                scannet_data = load_scannet_data(args, seq_len)
                if scannet_data is not None:
                    # Update model patch dimensions if method exists
                    if hasattr(model, 'update_patch_dimensions'):
                        model.update_patch_dimensions(scannet_data["patch_width"], scannet_data["patch_height"])
                    
                    # Forward pass to trigger hooks
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                        vgg_input = scannet_data["vgg_input"].to(args.device)
                        model(vgg_input)
                    
                    # Calculate token similarity for each global block layer
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
                                        "dataset": "ScanNet",
                                        "sequence_length": seq_len,
                                        "block_layer": block_layer,
                                        "token_similarity": similarity
                                    })
                    
                    # Clear tokens for next iteration
                    for key in tokens:
                        tokens[key] = None
                    
            except Exception as e:
                print(f"Error processing ScanNet dataset: {e}")
        
        # --- Memory Management ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # --- Remove Hooks ---
    for hook in hooks:
        hook.remove()
    
    # --- Save Results ---
    if results:
        # Save to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, "token_similarity_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Generate and save heatmaps for each dataset
        datasets = df["dataset"].unique()
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            if len(dataset_df) > 0:
                heatmap_path = os.path.join(args.output_dir, f"token_similarity_heatmap_{dataset}.png")
                generate_heatmap(dataset_df, block_layers, sequence_lengths, heatmap_path)
                print(f"Heatmap saved to {heatmap_path}")
    
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    # Rename argument to match expected variable name
    args.sevenscenes_data_root = args.__dict__.pop('7scenes_data_root', None)
    main(args)
