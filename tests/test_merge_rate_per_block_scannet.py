import os
import sys
import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, "eval"))

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)

# --- Global Configuration Variables ---
# These can be modified by the user before running the script
SEQUENCE_LENGTHS = [20]  # 输入帧的数量

# Per-block merge ratio configuration
# Important: Only GLOBAL blocks execute merging (frame blocks don't use global_merging parameter)
# The 4 ratios correspond to global_blocks[0], global_blocks[1], global_blocks[2], global_blocks[3]
# Remaining global blocks (4-23) will use merge_ratio=0.0 (no merging)
# 10 values for each of 4 global blocks = 10^4 = 10,000 combinations
PER_BLOCK_MERGE_RATIO_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_BLOCKS = 15

# Generate all 10^4 = 10,000 combinations of per-block merge ratios
def generate_all_per_block_combinations(start_from_config=None):
    """
    Generate all 10,000 combinations of 4 global-block merge ratios.
    Each global block (0-3) can have one of 10 values: [0.0, 0.1, 0.2, ..., 0.9]
    
    Note: These ratios apply to global_blocks[0-3] only. Frame blocks don't execute merging.
    
    Args:
        start_from_config: Optional tuple (r0, r1, r2, r3) to start from a specific configuration.
                          All combinations before this will be skipped.
                          e.g., (0.0, 0.5, 0.5, 0.5) to start from [0.0, 0.5, 0.5, 0.5]
    
    Returns: List of dicts {global_block_index: merge_ratio}
             e.g., {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3} means:
             - global_blocks[0] uses merge_ratio=0.0
             - global_blocks[1] uses merge_ratio=0.1
             - global_blocks[2] uses merge_ratio=0.2
             - global_blocks[3] uses merge_ratio=0.3
    """
    combinations = []
    should_include = start_from_config is None  # If no start config, include all
    
    for r0 in PER_BLOCK_MERGE_RATIO_VALUES:
        for r1 in PER_BLOCK_MERGE_RATIO_VALUES:
            for r2 in PER_BLOCK_MERGE_RATIO_VALUES:
                for r3 in PER_BLOCK_MERGE_RATIO_VALUES:
                    # Check if we've reached the start configuration
                    if not should_include and start_from_config == (r0, r1, r2, r3):
                        should_include = True
                    
                    if should_include:
                        config = {0: r0, 1: r1, 2: r2, 3: r3}
                        combinations.append(config)
    return combinations

# Generate combinations starting from [0.0, 0.5, 0.5, 0.5]
# Change to None to generate all 10,000 combinations from the beginning
START_FROM_CONFIG = (0.4, 0.2, 0.6, 0.9)  # (block_0, block_1, block_2, block_3)
PER_BLOCK_MERGE_RATIOS = generate_all_per_block_combinations(start_from_config=START_FROM_CONFIG)

NUM_TEST_SCENES = 15  # 测试的场景数量
def get_args_parser():
    parser = argparse.ArgumentParser("Test Merge Rate vs. Sequence Length for ScanNet", add_help=False)
    
    # Global variables that can be overridden by command line arguments
    parser.add_argument("--data_dir", type=Path, default="/home/hba/Documents/Dataset/ScanNet/scans/", 
                       help="Path to the ScanNet processed dataset root")
    parser.add_argument("--gt_ply_dir", type=Path, default="/home/hba/Documents/Dataset/ScanNet/scans/",
                       help="Path to the ScanNet raw scans directory")
    parser.add_argument("--ckpt_path", type=str, default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt", 
                       help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/home/hba/Documents/FastVGGT/tests/tests_result/ScanNet_merge_rateVs_seq_len", 
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    
    # Parameters for evaluation
    parser.add_argument("--depth_conf_thresh", type=float, default=1.0, 
                       help="Depth confidence threshold for filtering low confidence depth values")
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5, 
                       help="Maximum distance threshold in Chamfer Distance computation")
    parser.add_argument("--num_scenes", type=int, default=None, 
                       help="Maximum number of scenes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_per_block_combos", type=int, default=None,
                       help="Maximum number of per-block combinations to test (useful for quick testing). Default: None (test all 10,000)")
    
    return parser

def update_model_merge_ratio(model, merge_ratio, per_block_ratios=None):
    """
    Update the model's merge ratio for dynamic testing.
    
    Important: Only global blocks execute merging (frame blocks don't receive global_merging parameter).
    
    Args:
        model: VGGT model instance
        merge_ratio: Default merge ratio for blocks not specified in per_block_ratios (0.0-1.0)
        per_block_ratios: Optional dict {global_block_index: merge_ratio} for per-block settings
                         Keys should be 0-based indices for global_blocks (e.g., {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    """
    # If per_block_ratios is not provided or empty, use global merge_ratio for all blocks
    if not per_block_ratios:
        per_block_ratios = {}
    
    # Update the global merge ratio in the model/aggregator
    if hasattr(model, "merge_ratio"):
        model.merge_ratio = merge_ratio
    if hasattr(model, "aggregator"):
        model.aggregator.merge_ratio = merge_ratio
        
        # Enable merging from block 0 (global blocks will check their individual merge_ratio)
        model.aggregator.merging = 0
    
    # Frame blocks don't execute merging (no global_merging parameter in forward),
    # so we don't need to update their merge_ratio
    
    # Update ONLY global blocks with per-block merge ratios
    if hasattr(model, "aggregator") and hasattr(model.aggregator, "global_blocks"):
        for block_idx, block in enumerate(model.aggregator.global_blocks):
            # Use per-block ratio if specified for this global block index
            # Otherwise use the default merge_ratio (typically 0 for unused blocks)
            block_merge_ratio = per_block_ratios.get(block_idx, merge_ratio)
            
            if hasattr(block, 'attn'):
                block.attn.merge_ratio = block_merge_ratio
            if hasattr(block, 'self_attn'):
                block.self_attn.merge_ratio = block_merge_ratio

def process_scene(model, scene_data, seq_len, config_id, args, dtype):
    """
    Process a scene and return metrics.
    
    Args:
        model: VGGT model
        scene_data: Scene data dictionary
        seq_len: Sequence length
        config_id: Configuration identifier (for logging/temp directories)
        args: Command line arguments
        dtype: Data type
        
    Returns:
        Dictionary of metrics or None if processing failed
    """
    scene = scene_data["scene"]
    scene_dir = scene_data["scene_dir"]
    image_paths = scene_data["image_paths"]
    poses_gt = scene_data["poses_gt"]
    first_gt_pose = scene_data["first_gt_pose"]
    available_pose_frame_ids = scene_data["available_pose_frame_ids"]
    
    # Frame filtering based on sequence length
    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
        image_paths, available_pose_frame_ids, seq_len
    )
    
    if len(selected_image_paths) == 0:
        print(f"Warning: No images selected for scene {scene} with seq_len {seq_len}")
        return None
        
    # Get corresponding poses
    c2ws = poses_gt[selected_pose_indices] if poses_gt is not None else None
    
    # Load images
    images = load_images_rgb(selected_image_paths)
    if not images or len(images) < 3:
        print(f"Warning: Insufficient valid images for scene {scene}")
        return None
        
    # Prepare input for VGGT
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    
    # Update model attention layers with dynamic patch dimensions
    model.update_patch_dimensions(patch_width, patch_height)
    
    # Inference + Reconstruction with timing
    (
        extrinsic_np,
        intrinsic_np,
        all_world_points,
        all_point_colors,
        all_cam_to_world_mat,
        inference_time_ms,
    ) = infer_vggt_and_reconstruct(
        model,
        vgg_input,
        dtype,
        args.depth_conf_thresh,
        selected_image_paths,
        device=torch.device(args.device),
    )
    
    # Check if we got valid results
    if not all_cam_to_world_mat or not all_world_points:
        print(f"Warning: Failed to obtain valid results for scene {scene}")
        # Clean up on early return
        del images_array, vgg_input
        return None
        
    # Evaluate the scene
    output_scene_dir = Path(args.output_dir) / f"temp_{scene}_{seq_len}_{config_id}"
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = evaluate_scene_and_save(
        scene,
        c2ws,
        first_gt_pose,
        selected_frame_ids,
        all_cam_to_world_mat,
        all_world_points,
        output_scene_dir,
        args.gt_ply_dir,
        args.chamfer_max_dist,
        inference_time_ms,
        False,  # plot: disabled for faster large-scale testing
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_scene_dir)
    
    # Clean up temporary variables to free memory
    del images_array, vgg_input
    del extrinsic_np, intrinsic_np, all_point_colors
    
    if metrics is not None:
        result = {
            "chamfer_distance": float(metrics.get("chamfer_distance", 0.0)),
            "ate": float(metrics.get("ate", 0.0)),
            "are": float(metrics.get("are", 0.0)),
            "rpe_rot": float(metrics.get("rpe_rot", 0.0)),
            "rpe_trans": float(metrics.get("rpe_trans", 0.0)),
            "inference_time_ms": inference_time_ms,
        }
        return result
    
    return None

def main(args):
    """
    Main function to run the ScanNet per-block merge ratio experiment.
    
    Tests all 10,000 combinations of 4-block merge ratios (10^4 combinations).
    For each combination and sequence length, evaluates multiple ScanNet scenes
    and records average metrics incrementally to CSV.
    """
    # --- Setup for Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\nGenerated {len(PER_BLOCK_MERGE_RATIOS)} per-block merge ratio combinations")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Incremental CSV output setup
    csv_path = os.path.join(args.output_dir, "merge_rate_vs_seq_len_results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    # --- Scene Selection ---
    if args.num_scenes is not None:
        num_scenes = args.num_scenes
        scannet_scenes = get_all_scenes(args.data_dir, num_scenes)
    else:
        # Use the global NUM_TEST_SCENES variable
        num_scenes = NUM_TEST_SCENES
        scannet_scenes = get_all_scenes(args.data_dir, num_scenes)
    
    print(f"Testing on {len(scannet_scenes)} scenes from ScanNet dataset")
    
    # --- Load Scene Data ---
    scene_data_list = []
    for scene in scannet_scenes[:num_scenes]:  # Limit to specified number of scenes
        scene_dir = args.data_dir / f"{scene}"
        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        
        if (poses_gt is None or first_gt_pose is None or 
            available_pose_frame_ids is None or len(image_paths) == 0):
            print(f"Skipping scene {scene}: insufficient data")
            continue
            
        scene_data_list.append({
            "scene": scene,
            "scene_dir": scene_dir,
            "image_paths": image_paths,
            "poses_gt": poses_gt,
            "first_gt_pose": first_gt_pose,
            "available_pose_frame_ids": available_pose_frame_ids
        })
    
    if not scene_data_list:
        print("No valid scenes found to process")
        return
    
    print(f"Loaded data for {len(scene_data_list)} scenes")
    
    # --- Model Loading ---
    print(f"Loading model from: {args.ckpt_path}")
    model = VGGT(
        merging=0,  # Fixed at 0 for this test
        merge_ratio=0.9,  # Initial value, will be updated
        vis_attn_map=False,  # Disabled for faster testing
    )
    
    try:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    except FileNotFoundError:
        print(f"Checkpoint file not found at: {args.ckpt_path}")
        print("Please update the --ckpt_path argument.")
        return
    
    # Force use of bf16 data type
    device = torch.device(args.device)
    if device.type == "cuda":
        dtype = torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] < 8:
            print("WARNING: bfloat16 not supported on this GPU, falling back to float16")
            dtype = torch.float16
    else:
        dtype = torch.float32

    model = model.to(device).eval().to(dtype)
    
    # --- Experiment Loop ---
    # Use global variables for test parameters
    sequence_lengths = SEQUENCE_LENGTHS
    per_block_merge_ratios = PER_BLOCK_MERGE_RATIOS
    
    # Apply max_per_block_combos limit if specified
    if args.max_per_block_combos is not None:
        per_block_merge_ratios = per_block_merge_ratios[:args.max_per_block_combos]
        print(f"Limiting per-block combinations to {len(per_block_merge_ratios)}")
    
    for seq_len in sequence_lengths:
        print(f"\n=== Testing Sequence Length: {seq_len} ===")
        print(f"Testing {len(per_block_merge_ratios)} per-block merge ratio combinations")
        for per_block_idx, per_block_config in enumerate(per_block_merge_ratios):
            config_str = f"[{per_block_config[0]:.1f},{per_block_config[1]:.1f},{per_block_config[2]:.1f},{per_block_config[3]:.1f}]"
            
            # Show progress every 100 combinations
            if per_block_idx % 100 == 0:
                print(f"Progress: {per_block_idx}/{len(per_block_merge_ratios)} - Config {config_str}")
            
            # Update model with per-block config
            update_model_merge_ratio(model, 0, per_block_ratios=per_block_config)
            
            # Debug: Verify the first few configurations are applied correctly
            # Set to 0 to disable debug output, or use --verbose flag in production
            if per_block_idx < 5:
                print(f"  DEBUG: Applied config {config_str}")
                for i in range(4):
                    actual_ratio = model.aggregator.global_blocks[i].attn.merge_ratio
                    print(f"    global_blocks[{i}].attn.merge_ratio = {actual_ratio:.1f}")
            
            scene_metrics = []
            valid_scenes = 0
            
            try:
                for scene_data in tqdm(scene_data_list, desc=f"SeqLen {seq_len}, PerBlockCombo {per_block_idx}/{len(per_block_merge_ratios)}", leave=False):
                    metrics = process_scene(
                        model, scene_data, seq_len, per_block_idx, args, dtype
                    )
                    
                    if metrics is not None:
                        scene_metrics.append(metrics)
                        valid_scenes += 1
                    
                    # Clean up GPU cache after each scene to prevent memory accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"\nERROR processing combination {per_block_idx} with config {config_str}:")
                print(f"  Exception: {str(e)}")
                traceback.print_exc()
                print("  Skipping this configuration...\n")
                continue
            
            if valid_scenes > 0:
                # Calculate average metrics across scenes
                avg_metrics = {}
                for metric in ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"]:
                    values = [m[metric] for m in scene_metrics]
                    avg_metrics[f"avg_{metric}"] = np.mean(values)
                
                result_row = {
                    "sequence_length": seq_len,
                    "block_0_ratio": per_block_config[0],
                    "block_1_ratio": per_block_config[1],
                    "block_2_ratio": per_block_config[2],
                    "block_3_ratio": per_block_config[3],
                    "valid_scenes": valid_scenes,
                    **avg_metrics
                }

                # Incremental CSV write
                df = pd.DataFrame([result_row])
                df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
            
            # Clean up after processing this combination
            scene_metrics.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final cleanup after all experiments are done
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nResults saved to {csv_path}")

        # Print summary
        print("\n=== Summary of Results ===")
        print(df.to_string(index=False))
    else:
        print("No results to save")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("=== ScanNet Per-Block Merge Ratio Test Configuration ===")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"GT PLY directory: {args.gt_ply_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Number of scenes per config: {NUM_TEST_SCENES if args.num_scenes is None else args.num_scenes}")
    print(f"Sequence lengths to test: {SEQUENCE_LENGTHS} ({len(SEQUENCE_LENGTHS)} lengths)")
    print(f"Per-block merge ratio values: {PER_BLOCK_MERGE_RATIO_VALUES} ({len(PER_BLOCK_MERGE_RATIO_VALUES)} values per block)")
    print(f"Total per-block combinations: {len(PER_BLOCK_MERGE_RATIOS)} (10^4 = 10,000)")
    if args.max_per_block_combos:
        print(f"Limited to: {args.max_per_block_combos} combinations for this run")
    print("=" * 80)
    print()
    
    # Estimate total test time
    max_combos = args.max_per_block_combos if args.max_per_block_combos else len(PER_BLOCK_MERGE_RATIOS)
    total_configs = len(SEQUENCE_LENGTHS) * max_combos
    num_scenes = args.num_scenes if args.num_scenes else NUM_TEST_SCENES
    
    print(f"Total configurations to test: {total_configs}")
    print(f"Scenes per configuration: {num_scenes}")
    print(f"Estimated total scenes to process: {total_configs * num_scenes}")
    print(f"CSV output: {os.path.join(args.output_dir, 'merge_rate_vs_seq_len_results.csv')}")
    print()
    
    main(args)
