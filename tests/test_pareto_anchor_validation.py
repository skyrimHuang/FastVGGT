"""
自适应 Token 合并：Pareto 优化验证脚本
测试 4 组锚点配置（Conservative / Aggressive / Inverted / Optimal）
输出与 plot_pareto_results.py 兼容的 CSV 结果
"""

import os
import sys
import gc
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 将项目根目录加入 sys.path，便于绝对导入
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
)

# ============================================================================
# 配置：4 组锚点参数
# ============================================================================

ANCHOR_CONFIGS = {
    'Conservative': {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1},
    'Aggressive': {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9},
    'Inverted': {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3},
    'Optimal': {0: 0.4, 1: 0.3, 2: 0.8, 3: 0.9},
}

# ============================================================================
# 工具函数
# ============================================================================

def update_model_merge_ratio(model, merge_ratio, per_block_ratios=None):
    """
    动态更新模型的 merge ratio。

    注意：只有 global blocks 会执行跨帧合并；frame blocks 不执行该逻辑。

    Args:
        model: VGGT 模型实例
        merge_ratio: 默认合并率（当某层未在 per_block_ratios 中指定时使用）
        per_block_ratios: 可选字典 {global_block_index: merge_ratio}
    """
    if not per_block_ratios:
        per_block_ratios = {}
    
    if hasattr(model, "merge_ratio"):
        model.merge_ratio = merge_ratio
    if hasattr(model, "aggregator"):
        model.aggregator.merge_ratio = merge_ratio
        model.aggregator.merging = 0
    
    # 仅更新 global blocks 的逐层合并率
    if hasattr(model, "aggregator") and hasattr(model.aggregator, "global_blocks"):
        for block_idx, block in enumerate(model.aggregator.global_blocks):
            block_merge_ratio = per_block_ratios.get(block_idx, 0.0)
            
            if hasattr(block, 'attn'):
                block.attn.merge_ratio = block_merge_ratio
            if hasattr(block, 'self_attn'):
                block.self_attn.merge_ratio = block_merge_ratio


def compute_relative_metrics(metrics, baseline, tau=1.5):
    """
    按公式 3-3 ~ 3-6 计算相对退化指标。

    Args:
        metrics: 当前配置指标字典
        baseline: baseline 指标字典（merge_ratio 全为 0）
        tau: 截断阈值（默认 1.5）

    Returns:
        包含 E_i、time_ratio 与不同 alpha 分数的字典
    """
    # 公式 3-3：相对退化比率；公式 3-4：鲁棒截断
    geometric_metrics = ['chamfer_distance', 'ate', 'are', 'rpe_rot', 'rpe_trans']
    error_terms = []
    
    for metric_name in geometric_metrics:
        if baseline[metric_name] > 1e-6:
            x_tilde = metrics[metric_name] / baseline[metric_name]
            x_hat = min(x_tilde, tau)  # Eq 3-4
        else:
            x_hat = 1.0
        error_terms.append(x_hat)
    
    # 公式 3-5：五项几何误差等权平均
    E_i = sum(error_terms) / len(error_terms)
    
    # 耗时相对比率
    if baseline['inference_time_ms'] > 0:
        time_ratio = metrics['inference_time_ms'] / baseline['inference_time_ms']
    else:
        time_ratio = 1.0
    
    # 公式 3-6：不同 alpha 下的综合代价
    scores = {
        'score_alpha_050': 0.50 * E_i + 0.50 * time_ratio,
        'score_alpha_075': 0.75 * E_i + 0.25 * time_ratio,
        'score_alpha_100': 1.00 * E_i + 0.00 * time_ratio,
    }
    
    return {
        'E_i': float(E_i),
        'time_ratio': float(time_ratio),
        'score_alpha_050': float(scores['score_alpha_050']),
        'score_alpha_075': float(scores['score_alpha_075']),
        'score_alpha_100': float(scores['score_alpha_100']),
    }


def process_scene(model, scene_data, seq_len, config_idx, args, dtype):
    """
    处理单个场景并返回指标。
    
    Args:
        model: VGGT model
        scene_data: Scene data dictionary
        seq_len: Sequence length
        config_idx: Configuration index (for logging)
        args: Command line arguments
        dtype: Data type (torch.bfloat16 or torch.float32)
    
    Returns:
        Dictionary of metrics or None if processing failed
    """
    scene = scene_data["scene"]
    scene_dir = scene_data["scene_dir"]
    image_paths = scene_data["image_paths"]
    poses_gt = scene_data["poses_gt"]
    first_gt_pose = scene_data["first_gt_pose"]
    available_pose_frame_ids = scene_data["available_pose_frame_ids"]
    
    # 根据序列长度筛选帧
    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
        image_paths, available_pose_frame_ids, seq_len
    )
    
    if len(selected_image_paths) == 0:
        return None
        
    # 提取对应位姿
    c2ws = poses_gt[selected_pose_indices] if poses_gt is not None else None
    
    # 加载图像
    images = load_images_rgb(selected_image_paths)
    if not images or len(images) < 3:
        return None
        
    # 构造 VGGT 输入
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    
    # 根据分辨率更新 patch 维度
    model.update_patch_dimensions(patch_width, patch_height)
    
    # 推理 + 重建，并记录耗时
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
    
    # 结果有效性检查
    if not all_cam_to_world_mat or not all_world_points:
        del images_array, vgg_input
        return None
        
    # 评估当前场景
    output_scene_dir = Path(args.output_dir) / f"temp_{scene}_{seq_len}_{config_idx}"
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
        False,  # 关闭绘图以提升测试速度
    )
    
    # 清理临时目录
    import shutil
    shutil.rmtree(output_scene_dir)
    
    # 清理临时变量，释放显存/内存
    del images_array, vgg_input
    del extrinsic_np, intrinsic_np, all_point_colors
    
    if metrics is not None:
        result = {
            "chamfer_distance": float(metrics.get("chamfer_distance", 0.0)),
            "ate": float(metrics.get("ate", 0.0)),
            "are": float(metrics.get("are", 0.0)),
            "rpe_rot": float(metrics.get("rpe_rot", 0.0)),
            "rpe_trans": float(metrics.get("rpe_trans", 0.0)),
            "inference_time_ms": float(inference_time_ms),
        }
        return result
    
    return None


# ============================================================================
# 主实验流程
# ============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("Pareto Anchor Validation", add_help=False)
    
    parser.add_argument("--data_dir", type=Path, default="/home/hba/Documents/Dataset/ScanNet/scans/", 
                       help="Path to the ScanNet processed dataset root")
    parser.add_argument("--gt_ply_dir", type=Path, default="/home/hba/Documents/Dataset/ScanNet/scans/",
                       help="Path to the ScanNet raw scans directory")
    parser.add_argument("--ckpt_path", type=str, default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt", 
                       help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/home/hba/Documents/FastVGGT/tests/tests_result/pareto_analysis", 
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    
    parser.add_argument("--depth_conf_thresh", type=float, default=1.0, 
                       help="Depth confidence threshold")
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5, 
                       help="Maximum distance threshold in Chamfer Distance computation")
    parser.add_argument("--num_scenes", type=int, default=5,
                       help="Maximum number of scenes to evaluate")
    parser.add_argument("--seq_len", type=int, default=50,
                       help="Input frame count per scene")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser


def main(args):
    """
    主实验：在 ScanNet 场景上评估 4 组锚点配置。
    输出与 plot_pareto_results.py 兼容的 pareto_results_raw.csv。
    """
    
    # 可复现性设置
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 输出目录准备
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = Path(args.output_dir) / 'pareto_results_raw.csv'
    
    # CSV 表头定义
    csv_headers = [
        'config_name', 'config_idx', 'r0', 'r1', 'r2', 'r3',
        'scene',
        'E_i', 'time_ratio', 'score_alpha_050', 'score_alpha_075', 'score_alpha_100',
        'chamfer_distance', 'ate', 'are', 'rpe_rot', 'rpe_trans', 'inference_time_ms'
    ]
    
    # 若不存在则创建带表头的 CSV
    if not csv_path.exists():
        df_header = pd.DataFrame(columns=csv_headers)
        df_header.to_csv(csv_path, index=False)
        print(f"✓ Created CSV with headers: {csv_path}")
    
    # 场景选择
    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Testing on {len(scannet_scenes)} scenes from ScanNet dataset")
    
    # 加载场景数据
    scene_data_list = []
    for scene in scannet_scenes[:args.num_scenes]:
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
    
    print(f"Loaded data for {len(scene_data_list)} scenes\n")
    
    # 模型加载
    print(f"Loading model from: {args.ckpt_path}")
    model = VGGT(
        merging=0,
        merge_ratio=0.9,
        vis_attn_map=False,
    )
    
    try:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    except FileNotFoundError:
        print(f"Checkpoint file not found at: {args.ckpt_path}")
        return
    
    # 设备与数据类型
    device = torch.device(args.device)
    if device.type == "cuda":
        dtype = torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] < 8:
            print("WARNING: bfloat16 not supported on this GPU, falling back to float16")
            dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = model.to(device).eval().to(dtype)
    
    # ========================================================================
    # 阶段 1：计算 baseline（所有层 merge_ratio=0，不执行合并）
    # ========================================================================
    
    print("\n" + "="*80)
    print("阶段 1：计算 baseline（不合并）")
    print("="*80)
    
    baseline_config = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    update_model_merge_ratio(model, 0, per_block_ratios=baseline_config)
    
    baseline_metrics_dict = {}  # {scene_name: metrics}
    
    for scene_data in tqdm(scene_data_list, desc="计算 baseline 指标"):
        metrics = process_scene(model, scene_data, seq_len=args.seq_len, config_idx=-1, args=args, dtype=dtype)
        
        if metrics is not None:
            baseline_metrics_dict[scene_data['scene']] = metrics
            print(f"  ✓ {scene_data['scene']}: ATE={metrics['ate']:.4f}, Time={metrics['inference_time_ms']:.1f}ms")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存 baseline 指标
    baseline_json_path = Path(args.output_dir) / 'baseline_metrics.json'
    with open(baseline_json_path, 'w') as f:
        json.dump(baseline_metrics_dict, f, indent=2)
    print(f"\n已保存 baseline 指标: {baseline_json_path}\n")
    
    # ========================================================================
    # 阶段 2：评估 4 组锚点配置
    # ========================================================================
    
    print("="*80)
    print("阶段 2：评估 4 组锚点配置")
    print("="*80)
    
    for config_idx, (config_name, config_ratios) in enumerate(ANCHOR_CONFIGS.items()):
        
        print(f"\n[{config_idx+1}/{len(ANCHOR_CONFIGS)}] Testing {config_name}")
        print(f"  Config: r0={config_ratios[0]:.2f}, r1={config_ratios[1]:.2f}, r2={config_ratios[2]:.2f}, r3={config_ratios[3]:.2f}")
        
        # 应用当前配置
        update_model_merge_ratio(model, 0, per_block_ratios=config_ratios)
        
        # 校验配置是否生效
        for i in range(4):
            actual_ratio = model.aggregator.global_blocks[i].attn.merge_ratio
            expected_ratio = config_ratios[i]
            if abs(actual_ratio - expected_ratio) > 1e-6:
                print(f"  WARNING: Block {i} merge_ratio mismatch: expected {expected_ratio}, got {actual_ratio}")
        
        # 在所有场景上评估
        for scene_data in tqdm(scene_data_list, desc=f"  评估 {config_name}", leave=False):
            scene_name = scene_data['scene']
            
            # 若该场景没有 baseline，则跳过
            if scene_name not in baseline_metrics_dict:
                print(f"    Skipping {scene_name}: no baseline")
                continue
            
            try:
                # 计算当前配置指标
                metrics = process_scene(model, scene_data, seq_len=args.seq_len, config_idx=config_idx, args=args, dtype=dtype)
                
                if metrics is None:
                    print(f"    Failed to process {scene_name}")
                    continue
                
                # 计算相对指标（对比 baseline）
                baseline = baseline_metrics_dict[scene_name]
                rel_metrics = compute_relative_metrics(metrics, baseline, tau=1.5)
                
                # 组装结果行
                result_row = {
                    'config_name': config_name,
                    'config_idx': config_idx,
                    'r0': config_ratios[0],
                    'r1': config_ratios[1],
                    'r2': config_ratios[2],
                    'r3': config_ratios[3],
                    'scene': scene_name,
                    'E_i': rel_metrics['E_i'],
                    'time_ratio': rel_metrics['time_ratio'],
                    'score_alpha_050': rel_metrics['score_alpha_050'],
                    'score_alpha_075': rel_metrics['score_alpha_075'],
                    'score_alpha_100': rel_metrics['score_alpha_100'],
                    'chamfer_distance': metrics['chamfer_distance'],
                    'ate': metrics['ate'],
                    'are': metrics['are'],
                    'rpe_rot': metrics['rpe_rot'],
                    'rpe_trans': metrics['rpe_trans'],
                    'inference_time_ms': metrics['inference_time_ms'],
                }
                
                # 增量写入 CSV
                df = pd.DataFrame([result_row])
                df.to_csv(csv_path, mode='a', header=False, index=False)
                
            except Exception as e:
                import traceback
                print(f"    ERROR processing {scene_name}: {str(e)}")
                traceback.print_exc()
                continue
            
            # 清理 GPU 显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ========================================================================
    # 结果汇总
    # ========================================================================
    
    print("\n" + "="*80)
    print("实验完成")
    print("="*80)
    
    # 读取并输出汇总
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nResults saved to: {csv_path}")
        print(f"Total rows: {len(df)}")
        print("\nSummary by configuration:")
        summary = df.groupby('config_name').agg({
            'E_i': ['mean', 'std'],
            'time_ratio': ['mean', 'std'],
            'score_alpha_075': 'mean',
        }).round(4)
        print(summary)
    
    # 最终清理
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("="*80)
    print("=== Pareto 锚点验证：4 组配置测试 ===")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of scenes: {args.num_scenes}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Device: {args.device}")
    print("="*80)
    print()
    
    main(args)
