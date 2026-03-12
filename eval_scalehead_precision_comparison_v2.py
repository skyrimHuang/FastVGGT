#!/usr/bin/env python3
"""
ScaleHead 双目绝对尺度恢复精度验证（对应3.3.2节） - V2简化版

直接比较深度图，避免3D点云坐标系转换的复杂性
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vggt.models.vggt import VGGT
from eval.dataset_utils.kitti_stereo import KITTIStereoDataset


def _to_python_types(obj):
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_python_types(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ==================== 评估指标 ====================

def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> Dict[str, float]:
    """
    计算深度估计指标
    
    Args:
        pred_depth: 预测深度 [H, W]
        gt_depth: GT深度 [H, W]
        valid_mask: 有效像素mask [H, W]
    
    Returns:
        metrics: dict of float
    """
    metric_mask = (
        valid_mask
        & np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > 0)
        & (gt_depth >= min_depth)
        & (gt_depth <= max_depth)
    )

    pred = pred_depth[metric_mask]
    gt = gt_depth[metric_mask]
    
    if len(pred) == 0:
        return {
            'abs_rel': np.nan,
            'rmse': np.nan,
            'a1': np.nan,
        }
    
    # Absolute Relative Error
    abs_rel = np.mean(np.abs(pred - gt) / (gt + 1e-8))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    
    # Threshold accuracy (δ < 1.25)
    thresh = np.maximum(gt / (pred + 1e-8), pred / (gt + 1e-8))
    a1 = np.mean(thresh < 1.25)
    
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'a1': a1,
    }


def find_optimal_scale_ls(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    最小二乘法求最优尺度：s = (pred · gt) / (pred · pred)
    """
    numerator = np.sum(pred * gt)
    denominator = np.sum(pred * pred)
    if denominator < 1e-8:
        return 1.0
    return numerator / denominator


def evaluate_scalehead_precision(args):
    """
    主评估函数
    """
    device = args.device
    
    print("\n" + "="*80)
    print("  ScaleHead 双目绝对尺度恢复精度验证（3.3.2节）- V2简化版")
    print("="*80 + "\n")
    print(f"  评估过滤: min_disp>{args.min_disp}, depth_range=[{args.min_depth}, {args.max_depth}] m\n")
    
    # ========== 1. 加载模型 ==========
    print("[1/4] 加载模型...")
    model = VGGT(
        img_size=518,
        enable_camera=True,
        enable_depth=True,
        enable_scale_head=True,
        scale_head_kwargs={
            'hidden_activation': args.hidden_activation,
            'output_activation': args.output_activation,
            'log_scale_clip': args.log_scale_clip,
        },
        enable_point=False,
        enable_track=False,
        merging=False,  # 禁用token merging
        merge_ratio=0.0,
    )
    
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    print(f"  ✓ VGGT 加载自: {args.ckpt_path}")
    
    scale_head_state = torch.load(args.scale_head_ckpt, map_location="cpu")
    model.scale_head.load_state_dict(scale_head_state)
    print(f"  ✓ ScaleHead 加载自: {args.scale_head_ckpt}\n")
    
    model = model.to(device).eval()
    
    # ========== 2. 加载数据 ==========
    print("[2/4] 加载 KITTI 验证集...")
    val_indices = list(range(args.train_num, args.train_num + args.val_num))
    dataset = KITTIStereoDataset(
        data_dir=args.data_dir,
        indices=val_indices,
    )
    print(f"  ✓ 样本数: {len(dataset)}\n")
    
    # ========== 3. 逐样本评估 ==========
    print("[3/4] 逐样本评估...")
    results_per_sample = []
    skipped = 0
    
    for idx in tqdm(range(len(dataset)), desc="评估进度"):
        try:
            sample = dataset[idx]
            img_id = sample['metadata']['img_id']
            
            # 准备输入
            images_stereo = sample['images'].unsqueeze(0).to(device, dtype=torch.float32)  # [1, 2, 3, H, W]
            images_mono_left = images_stereo[:, :1, :, :, :]  # [1, 1, 3, H, W]
            
            calib = sample['calibration']
            baseline = calib['baseline']
            focal_scaled = calib['K_scaled'][0, 0]
            calib_features = torch.tensor([[baseline, focal_scaled]], dtype=torch.float32).to(device)
            
            gt_scale = sample['gt_scale'].item()
            H, W = images_stereo.shape[3], images_stereo.shape[4]
            model.update_patch_dimensions(W // 14, H // 14)
            
            disp_gt = sample['disparity']
            valid_disp = disp_gt > args.min_disp
            if valid_disp.sum() < 100:
                skipped += 1
                continue
            
            # 计算GT深度 (metric)
            depth_gt = np.zeros_like(disp_gt, dtype=np.float32)
            depth_gt[valid_disp] = baseline * focal_scaled / disp_gt[valid_disp]
            
            # ===== Baseline: 单目VGGT + 最优缩放因子 (Least Squares) =====
            t_baseline_start = time.time()
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16):
                    pred_baseline = model(images_mono_left, calibration_batch=None)
            t_baseline = (time.time() - t_baseline_start) * 1000
            
            depth_baseline_norm = pred_baseline['depth'][0, 0].detach().float().cpu().numpy().squeeze(-1)
            
            # 用最小二乘法找最优尺度
            pred_valid = depth_baseline_norm[valid_disp]
            gt_valid = depth_gt[valid_disp]
            scale_optimal = find_optimal_scale_ls(pred_valid, gt_valid)
            depth_baseline_metric = depth_baseline_norm * scale_optimal
            
            metrics_baseline = compute_depth_metrics(
                depth_baseline_metric,
                depth_gt,
                valid_disp,
                args.min_depth,
                args.max_depth,
            )
            
            # ===== Our Method: 双目VGGT + ScaleHead =====
            t_ours_start = time.time()
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16):
                    pred_ours = model(images_stereo, calibration_batch=calib_features)
            t_ours = (time.time() - t_ours_start) * 1000
            
            if 'scale_factor' not in pred_ours:
                skipped += 1
                continue
            
            pred_scale = pred_ours['scale_factor'][0, 0].item()
            scale_error_pct = abs(pred_scale - gt_scale) / (gt_scale + 1e-8) * 100
            
            depth_ours_norm = pred_ours['depth'][0, 0].detach().float().cpu().numpy().squeeze(-1)
            depth_ours_metric = depth_ours_norm * pred_scale
            
            metrics_ours = compute_depth_metrics(
                depth_ours_metric,
                depth_gt,
                valid_disp,
                args.min_depth,
                args.max_depth,
            )
            
            # 记录结果
            results_per_sample.append({
                'img_id': img_id,
                'gt_scale': gt_scale,
                'pred_scale': pred_scale,
                'scale_error_pct': scale_error_pct,
                'optimal_scale_baseline': scale_optimal,
                # Baseline metrics
                'baseline_abs_rel': metrics_baseline['abs_rel'],
                'baseline_rmse': metrics_baseline['rmse'],
                'baseline_a1': metrics_baseline['a1'],
                # Our method metrics
                'ours_abs_rel': metrics_ours['abs_rel'],
                'ours_rmse': metrics_ours['rmse'],
                'ours_a1': metrics_ours['a1'],
                # Timing
                'time_baseline_ms': t_baseline,
                'time_ours_ms': t_ours,
                'time_overhead_ms': t_ours - t_baseline,
            })

            del pred_baseline, pred_ours
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"  ⚠ Error processing sample {idx}: {e}")
            skipped += 1
            continue
    
    # ========== 4. 汇总与保存 ==========
    print("\n[4/4] 生成结果报告...")
    
    if len(results_per_sample) == 0:
        print("\n❌ 无有效结果，检查数据或模型路径\n")
        return
    
    df = pd.DataFrame(results_per_sample)
    
    # 保存详细结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_csv = output_dir / "scalehead_precision_detailed.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"\n  ✓ 详细结果: {detailed_csv}")
    
    # 计算统计汇总
    summary = {
        'num_samples': len(df),
        'num_skipped': skipped,
        # Scale prediction
        'scale_error_mean': df['scale_error_pct'].mean(),
        'scale_error_median': df['scale_error_pct'].median(),
        'scale_error_std': df['scale_error_pct'].std(),
        # Baseline metrics
        'baseline_abs_rel_mean': df['baseline_abs_rel'].mean(),
        'baseline_rmse_mean': df['baseline_rmse'].mean(),
        'baseline_a1_mean': df['baseline_a1'].mean(),
        # Our method metrics
        'ours_abs_rel_mean': df['ours_abs_rel'].mean(),
        'ours_rmse_mean': df['ours_rmse'].mean(),
        'ours_a1_mean': df['ours_a1'].mean(),
        # Relative improvement
        'abs_rel_gap_pct': (df['ours_abs_rel'].mean() - df['baseline_abs_rel'].mean()) / df['baseline_abs_rel'].mean() * 100,
        'rmse_gap_pct': (df['ours_rmse'].mean() - df['baseline_rmse'].mean()) / df['baseline_rmse'].mean() * 100,
        # Timing
        'time_overhead_mean_ms': df['time_overhead_ms'].mean(),
        'time_overhead_max_ms': df['time_overhead_ms'].max(),
    }
    
    summary_df = pd.DataFrame([
        {'metric': k, 'value': v}
        for k, v in summary.items()
    ])
    
    summary_csv = output_dir / "scalehead_precision_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ 统计汇总: {summary_csv}")
    
    # 打印关键结果
    print("\n" + "="*80)
    print("  📊 SUMMARY REPORT")
    print("="*80 + "\n")
    
    print("【数据统计】")
    print(f"  总样本数:       {summary['num_samples']}")
    print(f"  跳过样本:       {summary['num_skipped']}\n")
    
    print("【尺度预测精度】")
    print(f"  Mean Scale Error:   {summary['scale_error_mean']:.3f}%")
    print(f"  Median Scale Error: {summary['scale_error_median']:.3f}%")
    print(f"  Std Scale Error:    {summary['scale_error_std']:.3f}%\n")
    
    print("【深度估计精度 - Baseline (单目 + 最优尺度)】")
    print(f"  Abs Rel:  {summary['baseline_abs_rel_mean']:.4f}")
    print(f"  RMSE:     {summary['baseline_rmse_mean']:.4f} m")
    print(f"  δ<1.25:   {summary['baseline_a1_mean']*100:.2f}%\n")
    
    print("【深度估计精度 - Our Method (双目 + ScaleHead)】")
    print(f"  Abs Rel:  {summary['ours_abs_rel_mean']:.4f}")
    print(f"  RMSE:     {summary['ours_rmse_mean']:.4f} m")
    print(f"  δ<1.25:   {summary['ours_a1_mean']*100:.2f}%\n")
    
    print("【相对差距】")
    print(f"  Abs Rel Gap:  {summary['abs_rel_gap_pct']:+.2f}%")
    print(f"  RMSE Gap:     {summary['rmse_gap_pct']:+.2f}%\n")
    
    print("【计算耗时】")
    print(f"  Mean Overhead:       {summary['time_overhead_mean_ms']:.2f} ms")
    print(f"  Max Overhead:        {summary['time_overhead_max_ms']:.2f} ms\n")
    
    print("="*80)
    print("  ✨ 实验完成！结果已保存。")
    print("="*80 + "\n")
    
    # 保存JSON报告
    report = {
        'config': vars(args),
        'summary': summary,
        'per_sample_results': results_per_sample,
    }
    
    json_path = output_dir / "scalehead_precision_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python_types(report), f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 报告: {json_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ScaleHead 双目绝对尺度恢复精度验证 - V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="KITTI dataset root")
    parser.add_argument("--ckpt_path", type=str, 
                        default="ckpt/model_tracker_fixed_e20.pt",
                        help="VGGT backbone checkpoint")
    parser.add_argument("--scale_head_ckpt", type=str,
                        default="outputs/kitti_scale_head/checkpoints/scale_head_best.pt",
                        help="Trained ScaleHead checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/scalehead_precision_eval_v2",
                        help="Output directory for results")
    parser.add_argument("--train_num", type=int, default=120,
                        help="Number of training samples (to skip)")
    parser.add_argument("--val_num", type=int, default=50,
                        help="Number of validation samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--hidden_activation", type=str, default="gelu",
                        choices=["gelu", "silu", "relu"],
                        help="ScaleHead隐藏层激活函数")
    parser.add_argument("--output_activation", type=str, default="exp",
                        choices=["exp", "softplus"],
                        help="ScaleHead输出激活函数")
    parser.add_argument("--log_scale_clip", type=float, default=None,
                        help="log_scale裁剪阈值，None为不裁剪")
    parser.add_argument("--min_disp", type=float, default=1.0,
                        help="有效视差下限（像素），用于过滤超远点")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="评估最小深度（米）")
    parser.add_argument("--max_depth", type=float, default=80.0,
                        help="评估最大深度（米）")
    
    args = parser.parse_args()
    
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        args.device = "cpu"
    
    evaluate_scalehead_precision(args)


if __name__ == "__main__":
    main()
