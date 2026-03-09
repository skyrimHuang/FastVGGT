"""
7-Scenes长序列三策略对比评估

对应论文 3.3.1 节：基于空间特征的关键帧选择与特征复用有效性验证

三策略对比：
  A (全量推理 Baseline) - 不加过滤，直接输入所有帧到VGGT推理
  B (传统启发式过滤) - 使用像素差分启发式选择关键帧，但预提取特征（功能同无过滤）
  C (本文方法) - 使用DINOv2前置余弦相似度过滤，特征复用路径

关键特性：
  - 长序列输入（目标1000帧，自动降级到可用最大）
  - 每条结果即时写入CSV（防OOM丢数据）
  - 完整ATE/ARE/RPE轨迹误差计算与对齐
  - OOM专门标记："OOM@Stage" 标志具体哪个阶段爆显存

使用示例：
  python tests/eval_7scenes_3strategies_long.py \
    --scene_names chess,stairs \
    --requested_frames 1000 \
    --output_dir ./tests/eval_7scenes_long_3strategies
"""

import os
import sys
import gc
import re
import time
import argparse
import warnings
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))

from vggt.models.vggt import VGGT
from vggt.utils.keyframe_filter import KeyframeFilter
from vggt.utils.eval_utils import (
    load_poses as load_poses_orig,
    get_vgg_input_imgs,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    eval_trajectory,
    to_homogeneous,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

warnings.filterwarnings("ignore")


def load_poses(path):
    """
    Load poses from 7Scenes directory with 'frame-XXXXXX.pose.txt' naming.
    Wraps eval_utils.load_poses to handle filename parsing for 7Scenes format.
    """
    path = Path(path)
    pose_files = sorted(
        path.glob("frame-*.pose.txt"), 
        key=lambda x: int(re.search(r'frame-(\d+)', x.stem).group(1))
    )
    
    if len(pose_files) == 0:
        print(f"Warning: No pose files (frame-*.pose.txt) found in directory {path}")
        return None, None, None
    
    c2ws = []
    available_frame_ids = []
    
    for pose_file in pose_files:
        try:
            with open(pose_file, "r") as f:
                nums = [float(x) for x in f.read().strip().split()]
                pose = np.array(nums).reshape(4, 4)
                if not (np.isinf(pose).any() or np.isnan(pose).any()):
                    c2ws.append(pose)
                    # Extract frame index from 'frame-XXXXXX' format
                    match = re.search(r'frame-(\d+)', pose_file.stem)
                    if match:
                        frame_idx = int(match.group(1))
                        available_frame_ids.append(frame_idx)
        except Exception as e:
            print(f"Warning: Failed to load {pose_file}: {e}")
    
    if len(c2ws) == 0:
        return None, None, None
    
    c2ws = np.array(c2ws)
    first_c2w = c2ws[0]
    
    return c2ws, first_c2w, available_frame_ids


def extract_poses_from_predictions(predictions, vgg_input_shape):
    """从模型的predictions中提取相机poses
    
    Args:
        predictions: 模型输出的字典，包含pose_enc
        vgg_input_shape: VGG输入图像的shape (S, C, H, W)
        
    Returns:
        poses_est: numpy array of shape (N, 4, 4), 相机外参矩阵
    """
    if predictions is None or "pose_enc" not in predictions:
        return None
    
    # 从pose_enc提取extrinsic和intrinsic
    pose_enc = predictions["pose_enc"]
    image_h, image_w = vgg_input_shape[2], vgg_input_shape[3]
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (image_h, image_w))
    
    # 转换为numpy (extrinsic: [1, N, 3, 4])
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()  # [N, 3, 4]
    
    # 转换为齐次坐标 [N, 4, 4]
    poses_est = to_homogeneous(extrinsic_np)
    
    return poses_est


def normalize_keyframe_indices(indices):
    """将keyframe索引规范为一维np.int64数组。"""
    if indices is None:
        return None
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    indices = np.array(indices)
    if indices.ndim > 1:
        indices = indices.reshape(-1)
    return indices.astype(np.int64)


def compute_trajectory_metrics_safe(poses_est, poses_gt, frame_ids, 
                                    scene_info=None, strategy_info=None, keyframe_indices=None):
    """稳健计算ATE/ARE/RPE：优先align=True，失败时回退align=False。
    
    Args:
        poses_est: numpy array [N, 4, 4] - 估计pose
        poses_gt: numpy array [N, 4, 4] - 真值pose，**必须已按frame_ids对应**
        frame_ids: numpy array [N] - 对应的frame ID（用于validate）
        scene_info: dict - 场景信息（用于诊断）
        strategy_info: str - 策略名称（用于诊断）
        keyframe_indices: array - 关键帧索引（用于诊断）
        
    Returns:
        metrics dict 或 None
    """
    if poses_est is None or poses_gt is None or frame_ids is None:
        return None
    if len(poses_est) < 2 or len(poses_gt) < 2:
        return None
    
    # 验证shape匹配
    if len(poses_est) != len(poses_gt) or len(poses_est) != len(frame_ids):
        if strategy_info:
            print(f"⚠️  [{strategy_info}] Shape mismatch: poses_est={len(poses_est)}, poses_gt={len(poses_gt)}, frame_ids={len(frame_ids)}")
        return None
    
    # 关键检查：frame_ids应该是合法的
    if len(frame_ids) > 0:
        if np.any(frame_ids < 0) or np.any(np.isnan(frame_ids)):
            if strategy_info:
                print(f"⚠️  [{strategy_info}] Invalid frame IDs detected")
            return None

    use_align = len(poses_est) >= 3 and len(poses_gt) >= 3
    try:
        metrics, _, _ = eval_trajectory(poses_est, poses_gt, frame_ids, align=use_align)
        return metrics
    except Exception as e1:
        try:
            metrics, _, _ = eval_trajectory(poses_est, poses_gt, frame_ids, align=False)
            return metrics
        except Exception as e2:
            if strategy_info:
                print(f"⚠️  [{strategy_info}] eval_trajectory failed: {str(e1)[:30]} / {str(e2)[:30]}")
            return None


class KeyframeStrategyEvaluator:
    """三策略长序列评估器"""

    def __init__(self, model: VGGT, device: str, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.aggregator = model.aggregator

    def _try_infer_full(
        self,
        vgg_input: torch.Tensor,
        image_paths: List[str],
    ) -> Dict:
        """策略A：全量推理（无过滤）"""
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

            start_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(vgg_input)
            torch.cuda.synchronize(self.device)
            elapsed = time.time() - start_time

            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

            return {
                "success": True,
                "time_s": elapsed,
                "memory_mb": peak_mem,
                "kept_frames": len(vgg_input),
                "compression_ratio": 1.0,
                "error": None,
                "predictions": predictions,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            return {
                "success": False,
                "time_s": None,
                "memory_mb": peak_mem,
                "kept_frames": None,
                "compression_ratio": None,
                "error": "OOM@FullInference",
                "predictions": None,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "time_s": None,
                "memory_mb": None,
                "kept_frames": None,
                "compression_ratio": None,
                "error": f"InferenceError: {str(e)[:50]}",
                "predictions": None,
            }

    def _try_infer_heuristic_filter(
        self,
        vgg_input: torch.Tensor,
        image_paths: List[str],
        pixel_diff_threshold: float = 50.0,
    ) -> Dict:
        """策略B：像素差分启发式过滤（无特征复用）
        
        使用灰度像素差分判断帧是否为关键帧。
        区别于策略C的是，这里过滤出的帧仍需VGGT从头提特征。
        """
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

            # 转换为numpy进行像素差分（vgg_input: [S, 3, H, W]）
            vgg_np = vgg_input.detach().cpu().numpy()  # [S, 3, H, W]

            # 灰度化：0.299*R + 0.587*G + 0.114*B
            gray_frames = []
            for i in range(vgg_np.shape[0]):
                img_chw = vgg_np[i]  # [3, H, W]
                img_01 = np.transpose(img_chw, (1, 2, 0))  # [H, W, 3]
                gray = (
                    0.299 * img_01[:, :, 0]
                    + 0.587 * img_01[:, :, 1]
                    + 0.114 * img_01[:, :, 2]
                )  # [H, W]
                gray_frames.append(gray)

            # 关键帧选择：第一帧总是保留，其他帧与前一个关键帧比较
            keyframe_indices = [0]
            ref_gray = gray_frames[0]

            for i in range(1, len(gray_frames)):
                diff = np.abs(gray_frames[i] - ref_gray).mean()
                if diff > pixel_diff_threshold:
                    keyframe_indices.append(i)
                    ref_gray = gray_frames[i]

            # 确保至少保留2帧
            if len(keyframe_indices) < 2:
                keyframe_indices = [0, len(gray_frames) - 1]

            keyframe_indices = sorted(list(set(keyframe_indices)))
            filtered_vgg = vgg_input[keyframe_indices]  # [K, 3, H, W]
            filtered_image_paths = [image_paths[i] for i in keyframe_indices]

            # 执行推理（过滤后的帧）
            start_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(filtered_vgg)
            torch.cuda.synchronize(self.device)
            elapsed = time.time() - start_time

            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            compression_ratio = len(keyframe_indices) / len(vgg_input)

            return {
                "success": True,
                "time_s": elapsed,
                "memory_mb": peak_mem,
                "kept_frames": len(keyframe_indices),
                "compression_ratio": compression_ratio,
                "error": None,
                "predictions": predictions,
                "keyframe_indices": keyframe_indices,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            return {
                "success": False,
                "time_s": None,
                "memory_mb": peak_mem,
                "kept_frames": None,
                "compression_ratio": None,
                "error": "OOM@HeuristicFilter",
                "predictions": None,
                "keyframe_indices": None,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "time_s": None,
                "memory_mb": None,
                "kept_frames": None,
                "compression_ratio": None,
                "error": f"HeuristicError: {str(e)[:50]}",
                "predictions": None,
                "keyframe_indices": None,
            }

    def _try_infer_dino_filter_with_reuse(
        self,
        vgg_input: torch.Tensor,
        image_paths: List[str],
        threshold: float = 0.3,
    ) -> Dict:
        """策略C：DINOv2前置过滤 + 特征复用"""
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

            # 过滤阶段（逐帧提取）
            t0 = time.time()
            filter_model = KeyframeFilter(self.aggregator, threshold=threshold)
            images_input = vgg_input.unsqueeze(0)  # [1, S, 3, H, W]

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    filter_result = filter_model(images_input)

            t_filter = time.time() - t0

            filtered_images = filter_result["filtered_images"].squeeze(0)  # [K, 3, H, W]
            patch_tokens = filter_result.get("patch_tokens")
            kept_indices = filter_result.get("keyframe_indices", [])

            # 推理阶段（使用预计算特征）
            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(
                        filtered_images, precomputed_patch_tokens=patch_tokens
                    )
            torch.cuda.synchronize(self.device)
            t_inference = time.time() - t0

            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            compression_ratio = filter_result["stats"]["compression_ratio"]
            kept_frames = filter_result["stats"]["kept_frames"]

            return {
                "success": True,
                "time_s": t_filter + t_inference,
                "time_filter_s": t_filter,
                "time_inference_s": t_inference,
                "memory_mb": peak_mem,
                "kept_frames": kept_frames,
                "compression_ratio": compression_ratio,
                "error": None,
                "predictions": predictions,
                "keyframe_indices": kept_indices,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            return {
                "success": False,
                "time_s": None,
                "time_filter_s": None,
                "time_inference_s": None,
                "memory_mb": peak_mem,
                "kept_frames": None,
                "compression_ratio": None,
                "error": "OOM@DinoFilterOrInference",
                "predictions": None,
                "keyframe_indices": None,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "time_s": None,
                "time_filter_s": None,
                "time_inference_s": None,
                "memory_mb": None,
                "kept_frames": None,
                "compression_ratio": None,
                "error": f"DinoFilterError: {str(e)[:50]}",
                "predictions": None,
                "keyframe_indices": None,
            }


def get_7scenes_sequences(
    data_dir: str,
    scene_names: List[str],
    requested_frames: int,
) -> List[Dict]:
    """获取7Scenes序列列表，并记录实际可用帧数"""
    sequences = []

    for scene in scene_names:
        scene_dir = Path(data_dir) / scene

        if not scene_dir.exists():
            print(f"  ⚠️  场景 {scene} 不存在，跳过")
            continue

        # 取第一个序列
        seq_dirs = sorted(
            [d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("seq-")]
        )

        if not seq_dirs:
            print(f"  ⚠️  场景 {scene} 无 seq 子目录，跳过")
            continue

        seq_dir = seq_dirs[0]
        color_files = sorted(seq_dir.glob("*.color.png"))
        pose_dir = seq_dir

        actual_frames = len(color_files)
        if actual_frames < 10:
            print(f"  ⚠️  场景 {scene}/{seq_dir.name} 帧数不足 ({actual_frames}), 跳过")
            continue

        sequences.append(
            {
                "scene": scene,
                "sequence": seq_dir.name,
                "color_dir": str(seq_dir),
                "pose_dir": str(pose_dir),
                "requested_frames": requested_frames,
                "actual_frames": min(requested_frames, actual_frames),
                "available_frames": actual_frames,
            }
        )

    return sequences


def process_sequence(
    seq_info: Dict,
    evaluator: KeyframeStrategyEvaluator,
    output_csv_path: Path,
    csv_writer,
    csv_file,
) -> Tuple[bool, str]:
    """处理一个序列的三策略评估（每条结果即时写CSV）
    
    Returns:
        (success: bool, message: str)
    """
    scene = seq_info["scene"]
    sequence = seq_info["sequence"]
    color_dir = Path(seq_info["color_dir"])
    pose_dir = Path(seq_info["pose_dir"])
    requested_frames = seq_info["requested_frames"]
    actual_frames = seq_info["actual_frames"]
    available_frames = seq_info["available_frames"]

    try:
        # 加载图像和位姿
        color_files = sorted(color_dir.glob("*.color.png"))[:actual_frames]
        image_paths = [str(f) for f in color_files]

        images = load_images_rgb(image_paths)
        if not images or len(images) < 3:
            msg = f"{scene}/{sequence}: 无法加载足够的图像"
            print(f"    ⚠️  {msg}")
            return False, msg

        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        vgg_input = vgg_input.float().to(evaluator.device)

        # 加载位姿（用于ATE计算）
        poses_gt, first_gt_pose, available_frame_ids = load_poses(pose_dir)
        if poses_gt is None or len(available_frame_ids) == 0:
            msg = f"{scene}/{sequence}: 无法加载位姿数据"
            print(f"    ⚠️  {msg}")
            return False, msg

        available_frame_ids = np.array(available_frame_ids, dtype=np.int64)

        # 确保有足够的位姿匹配帧
        valid_frame_count = min(len(image_paths), len(available_frame_ids))
        if valid_frame_count < 3:
            msg = f"{scene}/{sequence}: 有效帧-位姿对数不足"
            print(f"    ⚠️  {msg}")
            return False, msg

        vgg_input = vgg_input[:valid_frame_count]
        image_paths = image_paths[:valid_frame_count]
        poses_gt = poses_gt[:valid_frame_count]
        available_frame_ids = available_frame_ids[:valid_frame_count]

        print(
            f"    ✓ 加载完成：{valid_frame_count} 帧、{vgg_input.shape[1]}x{vgg_input.shape[2]} 分辨率"
        )

        # ===== 策略A：全量推理 =====
        print(f"      [A] 全量推理...", end="", flush=True)
        result_a = evaluator._try_infer_full(vgg_input, image_paths)

        row_a = {
            "scene": scene,
            "sequence": sequence,
            "requested_frames": requested_frames,
            "actual_frames": valid_frame_count,
            "available_frames": available_frames,
            "strategy": "A_Full",
            "threshold_or_param": None,
            "kept_frames": result_a.get("kept_frames"),
            "redundancy_ratio": 1.0 - result_a.get("compression_ratio", 0.0)
            if result_a.get("compression_ratio") is not None
            else None,
            "compression_ratio": result_a.get("compression_ratio"),
            "total_time_s": result_a.get("time_s"),
            "fps": valid_frame_count / result_a.get("time_s")
            if result_a.get("time_s") is not None and result_a.get("time_s") > 0
            else None,
            "memory_mb": result_a.get("memory_mb"),
            "ate": None,
            "are": None,
            "rpe_trans": None,
            "rpe_rot": None,
            "success": result_a["success"],
            "error": result_a.get("error"),
        }

        if result_a["success"]:
            print(f" ✓ ({result_a['time_s']:.2f}s, {result_a['memory_mb']:.0f}MB)", end="")
            # 计算精度指标
            try:
                poses_est = extract_poses_from_predictions(
                    result_a.get("predictions"), vgg_input.shape
                )
                if poses_est is not None and len(poses_est) == len(poses_gt):
                    metrics = compute_trajectory_metrics_safe(
                        poses_est, poses_gt, available_frame_ids[: len(poses_est)]
                    )
                    if metrics is not None:
                        row_a["ate"] = metrics["ate"]
                        row_a["are"] = metrics["are"]
                        row_a["rpe_trans"] = metrics["rpe_trans"]
                        row_a["rpe_rot"] = metrics["rpe_rot"]
                        print(f", ATE={metrics['ate']:.3f}")
                    else:
                        print(", 精度计算跳过(评估失败)")
                else:
                    print(f", 精度计算跳过(poses不匹配)")
            except Exception as e:
                print(f", 精度计算失败: {str(e)[:30]}")
        else:
            print(f" ✗ {result_a['error']}")

        # 即时写入CSV
        csv_writer.writerow(row_a)
        csv_file.flush()

        # ===== 策略B：启发式过滤 =====
        print(f"      [B] 像素差分过滤...", end="", flush=True)
        result_b = evaluator._try_infer_heuristic_filter(
            vgg_input, image_paths, pixel_diff_threshold=0.04
        )

        row_b = {
            "scene": scene,
            "sequence": sequence,
            "requested_frames": requested_frames,
            "actual_frames": valid_frame_count,
            "available_frames": available_frames,
            "strategy": "B_Heuristic",
            "threshold_or_param": "pixel_diff=0.04",
            "kept_frames": result_b.get("kept_frames"),
            "redundancy_ratio": 1.0 - result_b.get("compression_ratio", 0.0)
            if result_b.get("compression_ratio") is not None
            else None,
            "compression_ratio": result_b.get("compression_ratio"),
            "total_time_s": result_b.get("time_s"),
            "fps": valid_frame_count / result_b.get("time_s")
            if result_b.get("time_s") is not None and result_b.get("time_s") > 0
            else None,
            "memory_mb": result_b.get("memory_mb"),
            "ate": None,
            "are": None,
            "rpe_trans": None,
            "rpe_rot": None,
            "success": result_b["success"],
            "error": result_b.get("error"),
        }

        if result_b["success"]:
            print(
                f" ✓ ({result_b['time_s']:.2f}s, {result_b['memory_mb']:.0f}MB, K={result_b['kept_frames']})",
                end="",
            )
            # 计算精度指标（按keyframe索引对齐GT）
            try:
                keyframe_indices = normalize_keyframe_indices(
                    result_b.get("keyframe_indices")
                )
                poses_est = extract_poses_from_predictions(
                    result_b.get("predictions"), vgg_input.shape
                )
                if (
                    poses_est is not None
                    and keyframe_indices is not None
                    and len(poses_est) == len(keyframe_indices)
                ):
                    # 验证索引范围
                    if np.any(keyframe_indices >= valid_frame_count) or np.any(keyframe_indices < 0):
                        print(", ✗ 关键帧索引越界")
                    else:
                        poses_gt_k = poses_gt[keyframe_indices]
                        frame_ids_k = available_frame_ids[keyframe_indices]
                        # 验证帧ID有效性
                        if len(frame_ids_k) == 0:
                            print(", ✗ 帧ID为空")
                        else:
                            metrics = compute_trajectory_metrics_safe(
                                poses_est, poses_gt_k, frame_ids_k
                            )
                            if metrics is not None:
                                row_b["ate"] = metrics["ate"]
                                row_b["are"] = metrics["are"]
                                row_b["rpe_trans"] = metrics["rpe_trans"]
                                row_b["rpe_rot"] = metrics["rpe_rot"]
                                print(f", ATE={metrics['ate']:.3f}")
                            else:
                                print(", 精度计算跳过(评估失败)")
                else:
                    print(f", ✗ poses或keyframes不匹配 (estim={len(poses_est) if poses_est is not None else None}, kf={len(keyframe_indices) if keyframe_indices is not None else None})")
            except Exception as e:
                print(f", 精度计算失败: {str(e)[:50]}")
        else:
            print(f" ✗ {result_b['error']}")

        # 即时写入CSV
        csv_writer.writerow(row_b)
        csv_file.flush()

        # ===== 策略C：DINOv2前置过滤 + 特征复用 =====
        for tau in [0.0005, 0.0007]:
            print(f"      [C] DINO过滤 (τ={tau})...", end="", flush=True)
            result_c = evaluator._try_infer_dino_filter_with_reuse(
                vgg_input, image_paths, threshold=tau
            )

            row_c = {
                "scene": scene,
                "sequence": sequence,
                "requested_frames": requested_frames,
                "actual_frames": valid_frame_count,
                "available_frames": available_frames,
                "strategy": "C_DINOReuse",
                "threshold_or_param": f"tau={tau}",
                "kept_frames": result_c.get("kept_frames"),
                "redundancy_ratio": 1.0 - result_c.get("compression_ratio", 0.0)
                if result_c.get("compression_ratio") is not None
                else None,
                "compression_ratio": result_c.get("compression_ratio"),
                "total_time_s": result_c.get("time_s"),
                "fps": valid_frame_count / result_c.get("time_s")
                if result_c.get("time_s") is not None and result_c.get("time_s") > 0
                else None,
                "memory_mb": result_c.get("memory_mb"),
                "ate": None,
                "are": None,
                "rpe_trans": None,
                "rpe_rot": None,
                "success": result_c["success"],
                "error": result_c.get("error"),
            }

            if result_c["success"]:
                print(
                    f" ✓ ({result_c['time_s']:.2f}s, {result_c['memory_mb']:.0f}MB, K={result_c['kept_frames']})",
                    end="",
                )
                # 计算精度指标（按DINO筛选keyframe索引对齐GT）
                try:
                    keyframe_indices = normalize_keyframe_indices(
                        result_c.get("keyframe_indices")
                    )
                    poses_est = extract_poses_from_predictions(
                        result_c.get("predictions"), vgg_input.shape
                    )
                    if (
                        poses_est is not None
                        and keyframe_indices is not None
                        and len(poses_est) == len(keyframe_indices)
                    ):
                        # 验证索引范围
                        if np.any(keyframe_indices >= valid_frame_count) or np.any(keyframe_indices < 0):
                            print(", ✗ 关键帧索引越界")
                        else:
                            poses_gt_k = poses_gt[keyframe_indices]
                            frame_ids_k = available_frame_ids[keyframe_indices]
                            # 验证帧ID有效性
                            if len(frame_ids_k) == 0:
                                print(", ✗ 帧ID为空")
                            else:
                                metrics = compute_trajectory_metrics_safe(
                                    poses_est, poses_gt_k, frame_ids_k,
                                    scene_info={"scene": scene, "seq": sequence},
                                    strategy_info=f"C_DINOReuse(τ={tau})",
                                    keyframe_indices=keyframe_indices
                                )
                                if metrics is not None:
                                    row_c["ate"] = metrics["ate"]
                                    row_c["are"] = metrics["are"]
                                    row_c["rpe_trans"] = metrics["rpe_trans"]
                                    row_c["rpe_rot"] = metrics["rpe_rot"]
                                    print(f", ATE={metrics['ate']:.3f}")
                                else:
                                    print(", 精度计算跳过(评估失败)")
                    else:
                        print(f", ✗ poses或keyframes不匹配 (estim={len(poses_est) if poses_est is not None else None}, kf={len(keyframe_indices) if keyframe_indices is not None else None})")
                except Exception as e:
                    print(f", 精度计算失败: {str(e)[:50]}")
            else:
                print(f" ✗ {result_c['error']}")

            # 即时写入CSV
            csv_writer.writerow(row_c)
            csv_file.flush()

        return True, f"{scene}/{sequence} 完成"

    except Exception as e:
        msg = f"{scene}/{sequence}: {str(e)[:100]}"
        print(f"    ❌ {msg}")
        return False, msg


def main(args):
    """主函数"""
    print("\n" + "=" * 80)
    print("7-Scenes 三策略长序列对比评估（含OOM即时保存）")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "keyframe_strategy_7scenes_long.csv"

    # 初始化CSV（即使文件已存在，也会覆盖）
    csv_fields = [
        "scene",
        "sequence",
        "requested_frames",
        "actual_frames",
        "available_frames",
        "strategy",
        "threshold_or_param",
        "kept_frames",
        "redundancy_ratio",
        "compression_ratio",
        "total_time_s",
        "fps",
        "memory_mb",
        "ate",
        "are",
        "rpe_trans",
        "rpe_rot",
        "success",
        "error",
    ]

    print(f"\n📋 配置:")
    print(f"   数据集路径: {args.data_dir}")
    print(f"   场景: {args.scene_names}")
    print(f"   请求帧数: {args.requested_frames}")
    print(f"   输出CSV: {csv_path}")

    # 加载模型
    print(f"\n🔄 加载模型...")
    device = args.device
    dtype = torch.float16

    model = VGGT(
        merging=25,
        merge_ratio=0.0,
        enable_point=True,
        enable_depth=True,
        enable_camera=True,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    print(f"   ✓ 模型加载完成 (设备: {device})")

    evaluator = KeyframeStrategyEvaluator(model, device, dtype)

    # 获取7Scenes序列列表
    print(f"\n📂 收集7Scenes序列...")
    scene_names = [s.strip() for s in args.scene_names.split(",")]
    sequences = get_7scenes_sequences(args.data_dir, scene_names, args.requested_frames)

    print(f"   找到 {len(sequences)} 个有效序列")

    if len(sequences) == 0:
        print("❌ 无有效序列，退出")
        return

    # 打开CSV文件并开始写入
    print(f"\n🚀 开始评估（含实时CSV保存）...\n")

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_file.flush()

        success_count = 0
        fail_count = 0

        for idx, seq_info in enumerate(sequences):
            print(f"  [{idx+1}/{len(sequences)}] {seq_info['scene']}/{seq_info['sequence']}")
            success, msg = process_sequence(
                seq_info, evaluator, csv_path, csv_writer, csv_file
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            # 间歇清理显存
            gc.collect()
            torch.cuda.empty_cache()

    # 加载并显示结果摘要
    print(f"\n" + "=" * 80)
    print("✅ 评估完成")
    print("=" * 80)

    print(f"\n📊 结果摘要:")
    print(f"   完成序列: {success_count}")
    print(f"   失败序列: {fail_count}")
    print(f"   CSV文件: {csv_path}")

    # 加载CSV并做基本统计
    try:
        df = pd.read_csv(csv_path)
        print(f"\n   总行数: {len(df)}")
        print(f"   成功: {df['success'].sum()}")
        print(f"   失败: {(~df['success']).sum()}")

        # 按策略统计
        print(f"\n   按策略分类:")
        for strategy in ["A_Full", "B_Heuristic", "C_DINOReuse"]:
            subset = df[df["strategy"] == strategy]
            if len(subset) > 0:
                success_pct = subset["success"].sum() / len(subset) * 100
                avg_time = subset[subset["success"]]["total_time_s"].mean()
                avg_mem = subset[subset["success"]]["memory_mb"].mean()
                avg_fps = subset[subset["success"]]["fps"].mean()
                avg_compression = subset[subset["success"]]["compression_ratio"].mean()

                print(f"     {strategy:15s}: {success_pct:5.1f}% 成功")
                if not np.isnan(avg_time):
                    print(f"       平均时间: {avg_time:.2f}s, FPS: {avg_fps:.2f}")
                    print(f"       平均显存: {avg_mem:.0f}MB, 压缩率: {avg_compression:.1%}")

        # OOM统计
        oom_errors = df[df["error"].astype(str).str.contains("OOM", na=False)]
        if len(oom_errors) > 0:
            print(f"\n   ⚠️  OOM错误统计:")
            oom_counts = oom_errors["error"].value_counts()
            for error_type, count in oom_counts.items():
                print(f"     {error_type}: {count}")

    except Exception as e:
        print(f"\n   ❌ 无法读取CSV: {e}")

    print(f"\n" + "=" * 80 + "\n")


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser("7-Scenes三策略长序列对比评估")

    parser.add_argument(
        "--scene_names",
        type=str,
        default="chess,stairs",
        help="7Scenes场景名称 (逗号分隔)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/hba/Documents/Dataset/7_scenes",
        help="7Scenes数据集路径",
    )
    parser.add_argument(
        "--requested_frames",
        type=int,
        default=1000,
        help="每个序列请求的帧数",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tests/eval_7scenes_long_3strategies",
        help="输出目录",
    )

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
