#!/usr/bin/env python3
"""
真实尺度三维重建推理脚本

使用 VGGT + 训练好的 ScaleHead 进行真实尺度（metric scale）三维重建。

核心流程：
  1. VGGT 对双目/多目图像进行推理，获得比例模型（up-to-scale）的深度/点云/相机位姿
  2. ScaleHead 利用立体特征和标定参数预测一个全局尺度因子 s
  3. 将深度 × s、点云 × s、平移 × s，得到真实尺度的三维重建

输出：
  - metric_reconstruction.ply  — 真实尺度点云（米为单位）
  - metric_poses.txt           — 真实尺度相机位姿
  - scale_info.json            — 预测尺度因子等元信息

用法示例（KITTI 立体对）：
  python infer_metric_scale.py \
    --left  /path/to/image_2/000000_10.png \
    --right /path/to/image_3/000000_10.png \
    --calib /path/to/calib_cam_to_cam/000000.txt \
    --scale_head_ckpt outputs/kitti_scale_head/checkpoints/scale_head_best.pt \
    --output_dir outputs/metric_reconstruction
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from eval.dataset_utils.kitti_calib import load_kitti_calibration


def load_and_preprocess_stereo(left_path: str, right_path: str, target_width: int = 518):
    """
    加载并预处理立体图像对。
    保持宽高比缩放到 target_width，高度对齐到 14 的倍数。

    Returns:
        images_tensor: [2, 3, H, W]  归一化到 [0,1]
        original_size: (W_orig, H_orig)
        target_size: (W_new, H_new)
    """
    to_tensor = TF.ToTensor()

    imgs = []
    for path in [left_path, right_path]:
        img = Image.open(path).convert("RGB")
        original_size = img.size  # (W, H)
        w, h = original_size

        new_width = target_width
        new_height = round(h * (new_width / w) / 14) * 14

        # 如果高度超过 518，裁切到 518
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        if new_height > 518:
            start_y = (new_height - 518) // 2
            img_arr = np.array(img)
            img_arr = img_arr[start_y:start_y + 518, :, :]
            img = Image.fromarray(img_arr)
            new_height = 518

        imgs.append(to_tensor(img))

    target_size = (new_width, new_height)
    images_tensor = torch.stack(imgs)  # [2, 3, H, W]
    return images_tensor, original_size, target_size


def save_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """保存点云为 PLY 格式"""
    valid = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                r, g, b = colors[i].astype(int)
                line += f" {r} {g} {b}"
            f.write(line + "\n")


@torch.no_grad()
def infer_metric_scale(args):
    device = args.device
    print(f"Device: {device}")

    # ========== 1. 加载模型 ==========
    print("Loading VGGT model...")
    model = VGGT(
        img_size=518,
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        enable_scale_head=True,
        merging=args.merging,
        merge_ratio=args.merge_ratio,
    )

    # 加载 backbone 权重
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    print(f"  Backbone loaded (missing: {len(incompat.missing_keys)}, unexpected: {len(incompat.unexpected_keys)})")

    # 加载 scale_head 权重
    scale_head_state = torch.load(args.scale_head_ckpt, map_location="cpu")
    model.scale_head.load_state_dict(scale_head_state)
    print(f"  ScaleHead loaded from {args.scale_head_ckpt}")

    model = model.to(device).eval()

    # ========== 2. 加载并预处理图像 ==========
    print("Loading stereo images...")
    images_tensor, orig_size, target_size = load_and_preprocess_stereo(
        args.left, args.right
    )
    print(f"  Original: {orig_size[0]}x{orig_size[1]} -> Target: {target_size[0]}x{target_size[1]}")

    # 更新 patch 维度
    patch_w = target_size[0] // 14
    patch_h = target_size[1] // 14
    model.update_patch_dimensions(patch_w, patch_h)

    # ========== 3. 加载 calibration ==========
    print("Loading calibration...")
    calib = load_kitti_calibration(args.calib, target_size=target_size)
    baseline = calib['baseline']
    focal_scaled = calib['K_scaled'][0, 0]
    print(f"  Baseline: {baseline:.4f} m, Focal (scaled): {focal_scaled:.2f} px")

    calib_features = torch.tensor([[baseline, focal_scaled]], dtype=torch.float32).to(device)

    # ========== 4. 推理 ==========
    print("Running inference...")
    images_input = images_tensor.unsqueeze(0).to(device, dtype=torch.float32)  # [1, 2, 3, H, W]

    torch.cuda.synchronize() if device != 'cpu' else None
    t0 = time.time()

    with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16):
        predictions = model(images_input, calibration_batch=calib_features)

    torch.cuda.synchronize() if device != 'cpu' else None
    t1 = time.time()
    print(f"  Inference time: {(t1-t0)*1000:.1f} ms")

    # ========== 5. 提取预测结果 ==========
    # 5a. 比例模型的深度
    depth_up_to_scale = predictions['depth'][0].detach().float().cpu().numpy()  # [S, H, W, 1]

    # 5b. 比例模型的相机位姿
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions['pose_enc'], (images_input.shape[3], images_input.shape[4])
    )
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()  # [S, 3, 4]
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()  # [S, 3, 3]

    # 5c. 预测的全局尺度因子
    scale_factor = predictions['scale_factor'][0, 0].item()
    print(f"  Predicted scale factor: {scale_factor:.4f}")

    # ========== 6. 应用尺度因子，获得真实尺度 ==========
    # 关键步骤：
    #   depth_metric = depth_up_to_scale * scale_factor
    #   translation_metric = translation_up_to_scale * scale_factor
    #
    # VGGT 输出的深度和平移在同一个比例模型空间中，
    # 乘以同一个尺度因子即可转换到真实尺度（米）

    depth_metric = depth_up_to_scale * scale_factor

    # 缩放相机平移（旋转不受尺度影响）
    extrinsic_metric = extrinsic_np.copy()
    extrinsic_metric[:, :3, 3] *= scale_factor  # 仅缩放平移部分

    # ========== 7. 反投影为真实尺度3D点云 ==========
    print("Generating metric-scale point cloud...")
    world_points = unproject_depth_map_to_point_map(
        depth_metric, extrinsic_metric, intrinsic_np
    )  # [S, H, W, 3]

    # 深度置信度过滤
    if 'depth_conf' in predictions:
        depth_conf = predictions['depth_conf'][0].detach().float().cpu().numpy()
        depth_mask = depth_conf >= args.depth_conf_thresh
    else:
        depth_mask = np.ones(depth_metric.shape[:3], dtype=bool)

    # 准备颜色
    images_np = images_tensor.detach().float().cpu().numpy()  # [2, 3, H, W]

    all_points = []
    all_colors = []
    for frame_idx in range(world_points.shape[0]):
        pts = world_points[frame_idx].reshape(-1, 3)
        mask = depth_mask[frame_idx].reshape(-1)
        valid = mask & ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)

        all_points.append(pts[valid])

        img_hwc = (np.transpose(images_np[frame_idx], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        colors_flat = img_hwc.reshape(-1, 3)
        all_colors.append(colors_flat[valid])

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    print(f"  Total points: {len(merged_points)}")

    # ========== 8. 保存结果 ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 点云
    ply_path = output_dir / "metric_reconstruction.ply"
    save_ply(str(ply_path), merged_points, merged_colors)
    print(f"  Saved: {ply_path}")

    # 相机位姿
    poses_path = output_dir / "metric_poses.txt"
    with open(poses_path, "w") as f:
        for i in range(extrinsic_metric.shape[0]):
            f.write(f"# Frame {i} (metric scale)\n")
            # 构建 4x4 矩阵
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :4] = extrinsic_metric[i]
            for row in pose_4x4:
                f.write(" ".join(f"{v:.8f}" for v in row) + "\n")
            f.write("\n")
    print(f"  Saved: {poses_path}")

    # 元信息
    info = {
        "scale_factor_predicted": scale_factor,
        "baseline_m": baseline,
        "focal_length_scaled_px": focal_scaled,
        "original_image_size": list(orig_size),
        "target_image_size": list(target_size),
        "num_points": len(merged_points),
        "inference_time_ms": (t1 - t0) * 1000,
        "depth_conf_threshold": args.depth_conf_thresh,
        "merging": args.merging,
        "merge_ratio": args.merge_ratio,
    }
    with open(output_dir / "scale_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {output_dir / 'scale_info.json'}")

    print("\nDone! Metric-scale reconstruction complete.")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  This means: VGGT_depth * {scale_factor:.4f} ≈ real depth in meters")


def main():
    parser = argparse.ArgumentParser(description="Metric-scale 3D reconstruction with VGGT + ScaleHead")

    # 图像输入
    parser.add_argument("--left", type=str, required=True, help="Left stereo image path")
    parser.add_argument("--right", type=str, required=True, help="Right stereo image path")
    parser.add_argument("--calib", type=str, required=True, help="KITTI calibration file path")

    # 模型
    parser.add_argument("--ckpt_path", type=str,
                        default="ckpt/model_tracker_fixed_e20.pt",
                        help="VGGT backbone checkpoint")
    parser.add_argument("--scale_head_ckpt", type=str,
                        default="outputs/kitti_scale_head/checkpoints/scale_head_best.pt",
                        help="Trained scale head checkpoint")

    # 推理参数
    parser.add_argument("--merging", type=int, default=0)
    parser.add_argument("--merge_ratio", type=float, default=0.9)
    parser.add_argument("--depth_conf_thresh", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/metric_reconstruction")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    infer_metric_scale(args)


if __name__ == "__main__":
    main()
