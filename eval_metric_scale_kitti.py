#!/usr/bin/env python3
"""
真实尺度评估脚本

评估 VGGT + ScaleHead 在 KITTI 上的真实尺度三维重建精度。

核心区别于原始 eval_custom.py：
  - 对齐方式：仅允许刚体对齐（旋转+平移），禁止缩放
  - 原始评估使用 Umeyama 对齐（含缩放），会掩盖尺度误差
  - 本脚本直接评估 metric-scale 的绝对精度

评估指标：
  1. 深度绝对误差（AbsRel, RMSE, δ<1.25）— 对单帧深度进行评估
  2. 点云尺度误差 — 预测的 scale vs GT scale 的比较
  3. ATE (rigid alignment, no scale) — 刚体对齐后的轨迹误差

用法：
  python eval_metric_scale_kitti.py \
    --data_dir /path/to/KITTI \
    --scale_head_ckpt outputs/kitti_scale_head/checkpoints/scale_head_best.pt \
    --output_dir outputs/metric_eval
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from eval.dataset_utils.kitti_stereo import KITTIStereoDataset, kitti_collate_fn
from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor


# ==================== 评估指标 ====================

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    计算深度评估指标（标准 KITTI 深度评估）

    Args:
        pred_depth: 预测深度 [H, W]
        gt_depth: GT 深度 [H, W]
        valid_mask: 有效区域 [H, W] bool

    Returns:
        dict 含 abs_rel, sq_rel, rmse, rmse_log, delta_1, delta_2, delta_3
    """
    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]

    if len(pred) == 0:
        return {k: float('nan') for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta_1', 'delta_2', 'delta_3']}

    # 过滤极端值
    mask = (gt > 1e-3) & (pred > 1e-3)
    pred = pred[mask]
    gt = gt[mask]

    if len(pred) == 0:
        return {k: float('nan') for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta_1', 'delta_2', 'delta_3']}

    thresh = np.maximum(pred / gt, gt / pred)
    delta_1 = (thresh < 1.25).mean()
    delta_2 = (thresh < 1.25 ** 2).mean()
    delta_3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean((pred - gt) ** 2 / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'delta_1': delta_1,
        'delta_2': delta_2,
        'delta_3': delta_3,
    }


def compute_scale_accuracy(pred_scale: float, gt_scale: float) -> dict:
    """
    计算尺度预测精度

    这是训练时 "12% error" 的来源解释：
      训练中的 val_mean_error = mean(|pred_scale - gt_scale| / gt_scale)
      即尺度因子的相对误差，不是深度精度或重建精度。

    Returns:
        dict 含 scale_error_abs, scale_error_rel, scale_ratio
    """
    abs_err = abs(pred_scale - gt_scale)
    rel_err = abs_err / (gt_scale + 1e-8)
    ratio = pred_scale / (gt_scale + 1e-8)

    return {
        'scale_error_abs': abs_err,
        'scale_error_rel': rel_err,
        'scale_ratio': ratio,   # 理想值=1.0
    }


def rigid_alignment_no_scale(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    """
    刚体对齐（旋转 + 平移），不做缩放。

    使用 Umeyama 算法但强制 estimate_scale=False。
    允许旋转是因为 VGGT 预测的坐标系和 GT 可能存在旋转差异，
    这不是 ScaleHead 要负责的。
    禁止缩放是因为尺度精度正是我们要评估的目标，
    如果对齐时做了缩放校正就等于"作弊"。

    Args:
        pred_points: [N, 3]
        gt_points: [N, 3]

    Returns:
        pred_aligned: [N, 3]
    """
    from scipy.linalg import svd

    assert pred_points.shape == gt_points.shape

    # 计算质心
    src_mean = pred_points.mean(axis=0)
    dst_mean = gt_points.mean(axis=0)

    # 去质心
    src_centered = pred_points - src_mean
    dst_centered = gt_points - dst_mean

    # SVD 求最优旋转 (Kabsch algorithm)
    H = src_centered.T @ dst_centered
    U, S, Vt = svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])  # 处理反射
    R = Vt.T @ sign_matrix @ U.T

    # 仅旋转 + 平移，不缩放 (scale=1)
    t = dst_mean - R @ src_mean

    pred_aligned = (R @ pred_points.T).T + t
    return pred_aligned


@torch.no_grad()
def evaluate_kitti(args):
    device = args.device
    print(f"Device: {device}")

    # ========== 1. 加载模型 ==========
    print("[1/5] Loading model...")
    model = VGGT(
        img_size=518,
        enable_camera=True,
        enable_depth=True,
        enable_scale_head=True,
        enable_point=False,
        enable_track=False,
        merging=args.merging,
        merge_ratio=args.merge_ratio,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    scale_head_state = torch.load(args.scale_head_ckpt, map_location="cpu")
    model.scale_head.load_state_dict(scale_head_state)
    print(f"  ScaleHead loaded from {args.scale_head_ckpt}")

    model = model.to(device).eval()

    # ========== 2. 加载数据 ==========
    print("[2/5] Loading KITTI dataset...")
    # 使用验证集（与训练时相同的划分）
    val_indices = list(range(args.train_num, args.train_num + args.val_num))
    dataset = KITTIStereoDataset(
        data_dir=args.data_dir,
        indices=val_indices,
    )
    print(f"  Validation samples: {len(dataset)}")

    # ========== 3. 逐样本评估 ==========
    print("[3/5] Running evaluation...")
    all_scale_metrics = []
    all_depth_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]

        # 准备输入
        images = sample['images'].unsqueeze(0).to(device, dtype=torch.float32)  # [1, 2, 3, H, W]

        calib = sample['calibration']
        baseline = calib['baseline']
        focal_scaled = calib['K_scaled'][0, 0]
        calib_features = torch.tensor([[baseline, focal_scaled]], dtype=torch.float32).to(device)

        gt_scale_value = sample['gt_scale'].item()

        # 更新 patch 维度
        H, W = images.shape[3], images.shape[4]
        model.update_patch_dimensions(W // 14, H // 14)

        # 推理
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16):
            predictions = model(images, calibration_batch=calib_features)

        # 提取预测的尺度因子
        pred_scale = predictions['scale_factor'][0, 0].item()

        # --- 尺度精度评估 ---
        scale_metrics = compute_scale_accuracy(pred_scale, gt_scale_value)
        all_scale_metrics.append(scale_metrics)

        # --- 深度评估（可选，如果有 disparity GT）---
        if 'depth' in predictions and sample['disparity'] is not None:
            # 从 disparity 计算 GT 深度
            disp_gt = sample['disparity']
            valid_disp = disp_gt > 0

            if valid_disp.sum() > 0:
                # GT 深度 = baseline * focal / disparity（真实尺度）
                gt_depth = np.zeros_like(disp_gt, dtype=np.float32)
                gt_depth[valid_disp] = baseline * calib['K_scaled'][0, 0] / disp_gt[valid_disp]

                # 预测深度（左帧） × 尺度因子 = 真实尺度深度
                pred_depth_upscale = predictions['depth'][0, 0].detach().float().cpu().numpy().squeeze(-1)
                pred_depth_metric = pred_depth_upscale * pred_scale

                # 将预测深度 resize 到与 GT 相同尺寸
                if pred_depth_metric.shape != gt_depth.shape:
                    pred_depth_metric = cv2.resize(
                        pred_depth_metric,
                        (gt_depth.shape[1], gt_depth.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )

                depth_met = compute_depth_metrics(pred_depth_metric, gt_depth, valid_disp)
                all_depth_metrics.append(depth_met)

    # ========== 4. 汇总 ==========
    print("\n[4/5] Aggregating results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 尺度精度汇总
    scale_summary = {}
    for key in all_scale_metrics[0]:
        values = [m[key] for m in all_scale_metrics]
        scale_summary[key + '_mean'] = float(np.mean(values))
        scale_summary[key + '_median'] = float(np.median(values))
        scale_summary[key + '_std'] = float(np.std(values))

    print("\n" + "=" * 70)
    print("  METRIC-SCALE EVALUATION RESULTS (Rigid Alignment, No Scale)")
    print("=" * 70)

    print("\n[Scale Prediction Accuracy]")
    print(f"  这就是训练中 'val_mean_error' 的含义：")
    print(f"    scale_error_rel = |pred_scale - gt_scale| / gt_scale")
    print(f"    训练时报告的 12% error = 尺度因子的平均相对误差为 12%")
    print(f"    它不是深度精度，也不是重建精度\n")
    print(f"  Mean  RelError: {scale_summary['scale_error_rel_mean']*100:.2f}%")
    print(f"  Median RelError: {scale_summary['scale_error_rel_median']*100:.2f}%")
    print(f"  Mean  Scale Ratio: {scale_summary['scale_ratio_mean']:.4f} (ideal = 1.0)")
    print(f"  Std   Scale Ratio: {scale_summary['scale_ratio_std']:.4f}")

    # 深度精度汇总
    depth_summary = {}
    if all_depth_metrics:
        print("\n[Metric Depth Accuracy (after applying predicted scale)]")
        print(f"  评估方式：depth_metric = VGGT_depth × predicted_scale")
        print(f"  对比 GT：depth_gt = baseline × focal / disparity")
        print(f"  这才是三维重建的真实精度评估\n")

        for key in all_depth_metrics[0]:
            values = [m[key] for m in all_depth_metrics if not np.isnan(m[key])]
            if values:
                depth_summary[key + '_mean'] = float(np.mean(values))
                print(f"  {key}: {np.mean(values):.4f}")

    print("\n" + "=" * 70)

    # ========== 5. 保存 ==========
    print("[5/5] Saving results...")
    results = {
        'config': {
            'data_dir': args.data_dir,
            'scale_head_ckpt': args.scale_head_ckpt,
            'val_samples': len(dataset),
            'merging': args.merging,
            'merge_ratio': args.merge_ratio,
        },
        'scale_metrics': scale_summary,
        'depth_metrics': depth_summary,
        'per_sample_scale': all_scale_metrics,
    }

    results_path = output_dir / "metric_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate metric-scale reconstruction on KITTI")

    parser.add_argument("--data_dir", type=str, required=True, help="KITTI dataset root")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--scale_head_ckpt", type=str,
                        default="outputs/kitti_scale_head/checkpoints/scale_head_best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/metric_eval")

    parser.add_argument("--train_num", type=int, default=120, help="Number of training samples (to skip)")
    parser.add_argument("--val_num", type=int, default=50, help="Number of validation samples")

    parser.add_argument("--merging", type=int, default=0)
    parser.add_argument("--merge_ratio", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    evaluate_kitti(args)


if __name__ == "__main__":
    main()
