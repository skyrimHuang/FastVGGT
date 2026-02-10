import os
import sys
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
SEQUENCE_LENGTHS = [5, 10, 20 , 30, 50, 100, 150]  # 输入帧的数量
MERGE_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_TEST_SCENES = 30  # 测试的场景数量
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
    
    # Parameters from eval_scannet.py with default values
    parser.add_argument("--merging", type=int, default=0, help="Merging parameter")
    parser.add_argument("--plot", type=bool, default=True, help="Generate plots")
    parser.add_argument("--depth_conf_thresh", type=float, default=1.0, 
                       help="Depth confidence threshold for filtering low confidence depth values")
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5, 
                       help="Maximum distance threshold in Chamfer Distance computation")
    parser.add_argument("--num_scenes", type=int, default=None, 
                       help="Maximum number of scenes to evaluate")
    parser.add_argument("--vis_attn_map", action="store_true", 
                       help="Whether to visualize attention maps during inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser

def update_model_merge_ratio(model, merge_ratio):
    """
    Update the model's merge ratio for dynamic testing.
    
    Args:
        model: VGGT model instance
        merge_ratio: Merge ratio to set (0.0-1.0)
    """
    # Update the merge ratio in the model/aggregator
    if hasattr(model, "merge_ratio"):
        model.merge_ratio = merge_ratio
    if hasattr(model, "aggregator"):
        model.aggregator.merge_ratio = merge_ratio
        
        # Enable or disable merging based on merge_ratio
        if merge_ratio > 0:
            # Enable merging from block 0
            model.aggregator.merging = 0
        else:
            # Disable merging by setting threshold higher than total blocks
            model.aggregator.merging = model.aggregator.depth + 1
    
    # Update attention layers with the new merge ratio
    for block in model.aggregator.frame_blocks:
        if hasattr(block, 'attn'):
            block.attn.merge_ratio = merge_ratio
    for block in model.aggregator.global_blocks:
        if hasattr(block, 'attn'):
            block.attn.merge_ratio = merge_ratio

def process_scene(model, scene_data, seq_len, merge_ratio, args, dtype):
    """
    Process a scene and return metrics.
    
    Args:
        model: VGGT model
        scene_data: Scene data dictionary
        seq_len: Sequence length
        merge_ratio: Merge ratio
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
        return None
        
    # Evaluate the scene
    output_scene_dir = Path(args.output_dir) / f"temp_{scene}_{seq_len}_{merge_ratio}"
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
        args.plot,
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_scene_dir)
    
    if metrics is not None:
        return {
            "chamfer_distance": float(metrics.get("chamfer_distance", 0.0)),
            "ate": float(metrics.get("ate", 0.0)),
            "are": float(metrics.get("are", 0.0)),
            "rpe_rot": float(metrics.get("rpe_rot", 0.0)),
            "rpe_trans": float(metrics.get("rpe_trans", 0.0)),
            "inference_time_ms": inference_time_ms,
        }
    
    return None

def main(args):
    """
    Main function to run the ScanNet merge rate vs sequence length experiment.
    """
    # --- Setup for Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
        merging=args.merging,
        merge_ratio=0.9,  # Initial value, will be updated
        vis_attn_map=args.vis_attn_map,
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
    merge_ratios = MERGE_RATIOS
    
    for seq_len in sequence_lengths:
        print(f"\n=== Testing Sequence Length: {seq_len} ===")
        
        for merge_ratio in merge_ratios:
            print(f"\n--- Testing Merge Ratio: {merge_ratio} ---")
            
            # Update model with current merge ratio
            update_model_merge_ratio(model, merge_ratio)
            
            scene_metrics = []
            valid_scenes = 0
            
            for scene_data in tqdm(scene_data_list, desc=f"SeqLen {seq_len}, Merge {merge_ratio}"):
                metrics = process_scene(
                    model, scene_data, seq_len, merge_ratio, args, dtype
                )
                
                if metrics is not None:
                    scene_metrics.append(metrics)
                    valid_scenes += 1
            
            if valid_scenes > 0:
                # Calculate average metrics across scenes
                avg_metrics = {}
                for metric in ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"]:
                    values = [m[metric] for m in scene_metrics]
                    avg_metrics[f"avg_{metric}"] = np.mean(values)
                    avg_metrics[f"std_{metric}"] = np.std(values)
                
                print(f"Processed {valid_scenes} scenes successfully")
                print(f"Average Chamfer Distance: {avg_metrics['avg_chamfer_distance']:.4f}")
                print(f"Average ATE: {avg_metrics['avg_ate']:.4f}")
                print(f"Average ARE: {avg_metrics['avg_are']:.4f}")
                print(f"Average RPE Rotation: {avg_metrics['avg_rpe_rot']:.4f}")
                print(f"Average RPE Translation: {avg_metrics['avg_rpe_trans']:.4f}")
                print(f"Average Inference Time: {avg_metrics['avg_inference_time_ms']:.2f}ms")
                
                result_row = {
                    "sequence_length": seq_len,
                    "merge_ratio": merge_ratio,
                    "valid_scenes": valid_scenes,
                    **avg_metrics
                }

                # Incremental CSV write
                df = pd.DataFrame([result_row])
                df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
            else:
                print(f"No scenes processed successfully for this configuration")
            
    
    # --- Save Results to CSV ---
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
    print("=== ScanNet Merge Rate vs Sequence Length Test Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"GT PLY directory: {args.gt_ply_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Number of scenes: {NUM_TEST_SCENES if args.num_scenes is None else args.num_scenes}")
    print(f"Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Merge ratios: {MERGE_RATIOS}")
    print("=" * 60)
    
    main(args)

"""

1. sequence_length（序列长度）
输入到VGGT模型的图像帧数量
代码中定义为 SEQUENCE_LENGTHS = [5]
表示每个场景使用多少帧图像进行重建

2. merge_ratio（token合并率）
Vision Transformer中token合并的比例，范围0.0-1.0
值越大，合并的token越多，计算量越小但可能损失精度
merge_ratio=0 表示不进行token合并

3. valid_scenes（有效场景数）
在该配置下成功处理的ScanNet场景数量
有些场景可能因数据不足或处理失败而被跳过

4. avg_chamfer_distance（平均Chamfer距离）
衡量预测点云与真值点云之间的相似度
计算双向最近点距离的平均值（预测→真值 + 真值→预测）
值越小越好，表示重建的3D点云更接近真实场景

5. std_chamfer_distance（Chamfer距离标准差）
表示不同场景之间Chamfer距离的波动程度
值越小说明模型在不同场景上表现越稳定

6. avg_ate（平均绝对轨迹误差, Absolute Trajectory Error）
衡量估计相机轨迹与真值轨迹之间的整体偏差（经过对齐后）
计算对应帧之间的欧氏距离RMSE
值越小越好，单位通常为米

7. avg_are（平均绝对旋转误差, Absolute Rotation Error）
衡量估计相机旋转与真值旋转之间的角度偏差
值越小越好，单位为度或弧度

8. avg_rpe_rot（平均相对位姿误差-旋转部分）
衡量连续帧之间的相对旋转误差
评估短期运动估计的旋转精度
值越小越好

9. avg_rpe_trans（平均相对位姿误差-平移部分）
衡量连续帧之间的相对平移误差
评估短期运动估计的平移精度
值越小越好

10. std_ate, std_are, std_rpe_rot, std_rpe_trans
对应指标在不同场景上的标准差
反映模型稳定性

11. avg_inference_time_ms（平均推理时间）
VGGT模型对一个序列进行推理的平均耗时
单位：毫秒（ms）
值越小越好，反映模型计算效率

12. std_inference_time_ms（推理时间标准差）
不同场景推理时间的波动
"""
