"""
对比过滤前后的重建精度评估脚本。

支持7Scenes和ScanNet数据集，分别评估：
1. 未过滤情况下的重建精度（baseline）
2. 使用keyframe_filter过滤后的重建精度
3. 计算精度差异和性能提升

在OOM情况下优雅地处理和显示。

使用示例:
    # 7Scenes评估
    python eval/eval_reconstruction_comparison.py \
        --dataset_type 7scenes \
        --data_dir /home/hba/Documents/Dataset/7_scenes \
        --scene_names chess fire heads office \
        --thresholds 0.3,0.5,0.7 \
        --output_dir ./tests/eval_reconstruction

    # ScanNet评估
    python eval/eval_reconstruction_comparison.py \
        --dataset_type scannet \
        --data_dir /home/hba/Documents/Dataset/ScanNet/scans \
        --num_scenes 10 \
        --thresholds 0.3,0.5 \
        --output_dir ./tests/eval_reconstruction
"""

import os
import sys
import json
import time
import argparse
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

from vggt.models.vggt import VGGT
from vggt.utils.keyframe_filter import KeyframeFilter
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


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        "Reconstruction Quality Comparison: Filtered vs No-Filter"
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["7scenes", "scannet"],
        default="7scenes",
        help="选择数据集类型",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/hba/Documents/Dataset/7_scenes",
        help="数据集路径",
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=str,
        default="/home/hba/Documents/Dataset/ScanNet/scans",
        help="地面真值PLY文件路径（ScanNet）",
    )
    
    # 场景选择参数
    parser.add_argument(
        "--scene_names",
        type=str,
        default=None,
        help="7Scenes场景名称（逗号分隔），如 chess,fire,heads,office",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="ScanNet场景数量上限",
    )
    
    # 过滤参数
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.5,0.7",
        help="过滤阈值（逗号分隔）",
    )
    
    # 模型参数
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--merging",
        type=int,
        default=None,
        help="是否使用token merging",
    )
    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.9,
        help="Token merge比例 (0.0-1.0)",
    )
    
    # 评估参数
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=1.0,
        help="深度置信度阈值",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Chamfer距离最大值",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=100,
        help="每个场景的最大帧数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备",
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tests/eval_reconstruction",
        help="输出目录",
    )
    
    return parser


class ReconstructionComparator:
    """对比过滤前后重建精度的评估器"""
    
    def __init__(self, model: VGGT, aggregator, device: str, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.aggregator = aggregator
        
    def _try_infer_no_filter(
        self,
        vgg_input: torch.Tensor,
        image_paths: List[str],
    ) -> Dict:
        """不使用过滤的推理（baseline）"""
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            
            extrinsic_np, intrinsic_np, all_points, all_colors, all_cam_to_world_mat, inference_time = (
                infer_vggt_and_reconstruct(
                    self.model,
                    vgg_input,
                    self.dtype,
                    depth_conf_thresh=1.0,
                    image_paths=image_paths,
                )
            )
            
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            
            return {
                "success": True,
                "extrinsic": extrinsic_np,
                "intrinsic": intrinsic_np,
                "points": all_points,
                "colors": all_colors,
                "poses": all_cam_to_world_mat,
                "inference_time": inference_time,
                "memory": peak_mem,
                "error": None,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "error": "OOM",
                "inference_time": None,
                "memory": None,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "error": f"InferenceError: {str(e)[:50]}",
                "inference_time": None,
                "memory": None,
            }

    def _try_infer_with_filter(
        self,
        vgg_input: torch.Tensor,
        image_paths: List[str],
        threshold: float,
    ) -> Dict:
        """使用过滤的推理"""
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            
            # 创建过滤器和执行过滤
            filter = KeyframeFilter(self.aggregator, threshold=threshold)
            
            with torch.no_grad():
                # vgg_input: [S, 3, H, W]
                # 需要reshape为[1, S, 3, H, W]以符合过滤器输入格式
                images_input = vgg_input.unsqueeze(0)  # [1, S, 3, H, W]
                
                # 执行过滤：使用autocast以兼容float16的模型权重
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    filter_result = filter(images_input)
                
                filtered_images = filter_result["filtered_images"].squeeze(0)  # [S', 3, H, W]
                patch_tokens = filter_result.get("patch_tokens")
                
                # 获取被保留的帧索引以对应image_paths
                kept_indices = filter_result.get("keyframe_indices", None)
                filtered_image_paths = None
                if kept_indices is not None and len(kept_indices) > 0:
                    # kept_indices是List[List[int]]，对于B=1的情况取第一个
                    filtered_image_paths = [image_paths[i] for i in kept_indices[0]]
                else:
                    filtered_image_paths = image_paths
                
                # 执行推理（filtered_images已经是原始的float32）
                extrinsic_np, intrinsic_np, all_points, all_colors, all_cam_to_world_mat, inference_time = (
                    infer_vggt_and_reconstruct(
                        self.model,
                        filtered_images,
                        self.dtype,
                        depth_conf_thresh=1.0,
                        image_paths=filtered_image_paths,
                    )
                )
            
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            compression_ratio = len(filtered_images) / len(vgg_input)
            
            return {
                "success": True,
                "extrinsic": extrinsic_np,
                "intrinsic": intrinsic_np,
                "points": all_points,
                "colors": all_colors,
                "poses": all_cam_to_world_mat,
                "inference_time": inference_time,
                "memory": peak_mem,
                "compression_ratio": compression_ratio,
                "kept_frames": len(filtered_images),
                "error": None,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "error": "OOM",
                "inference_time": None,
                "memory": None,
                "compression_ratio": None,
                "kept_frames": None,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "success": False,
                "error": f"FilterError: {str(e)[:50]}",
                "inference_time": None,
                "memory": None,
                "compression_ratio": None,
                "kept_frames": None,
            }


def evaluate_7scenes(
    data_dir: str,
    scene_names: List[str],
    thresholds: List[float],
    comparator: ReconstructionComparator,
    output_dir: str,
    input_frame: int,
) -> pd.DataFrame:
    """评估7Scenes数据集
    
    7Scenes格式: 
        data_dir/scene_name/seq-01/frame-000001.color.png
                                  /frame-000001.pose.txt
    """
    print(f"\n📊 开始评估 7Scenes 数据集...")
    
    results = []
    
    for scene_name in tqdm(scene_names, desc="Scenes"):
        scene_dir = Path(data_dir) / scene_name
        
        if not scene_dir.exists():
            print(f"  ⚠️ 场景 {scene_name} 不存在，跳过")
            continue
        
        # 遍历该场景下的所有序列
        seq_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("seq-")])
        
        if not seq_dirs:
            print(f"  ⚠️ 场景 {scene_name} 没有seq子目录，跳过")
            continue
        
        # 只用第一个序列进行评估
        seq_dir = seq_dirs[0]
        print(f"\n  📍 处理场景: {scene_name} / {seq_dir.name}")
        
        try:
            # 列出color文件
            color_files = sorted(seq_dir.glob("*.color.png"))
            if len(color_files) < 3:
                print(f"    ⚠️ 帧数不足 ({len(color_files)})")
                continue
            
            # 限制帧数
            color_files = color_files[:input_frame]
            image_paths = [str(f) for f in color_files]
            
            # 加载图像
            images = load_images_rgb(image_paths)
            if not images or len(images) < 3:
                print(f"    ⚠️ 有效图像数量不足")
                continue
            
            images_array = np.stack(images)
            vgg_input, _, _ = get_vgg_input_imgs(images_array)  # 返回tensor
            
            # vgg_input已经是tensor [S, 3, H, W]，无需再转换
            vgg_input = vgg_input.float().to(comparator.device)
            
            print(f"    输入: {len(image_paths)} 帧, 分辨率 {vgg_input.shape}")
            
            # ===== 无过滤推理 =====
            print(f"    1️⃣  无过滤推理...", end="", flush=True)
            no_filter_result = comparator._try_infer_no_filter(vgg_input, image_paths)
            
            if no_filter_result["success"]:
                print(f" ✓ (时间: {no_filter_result['inference_time']:.2f}ms, 显存: {no_filter_result['memory']:.0f}MB)")
                results.append({
                    "dataset": "7scenes",
                    "scene": scene_name,
                    "method": "no_filter",
                    "threshold": None,
                    "frames": len(image_paths),
                    "success": True,
                    "inference_time_ms": no_filter_result["inference_time"],
                    "memory_mb": no_filter_result["memory"],
                    "compression_ratio": 1.0,
                    "error": None,
                })
            else:
                print(f" ❌ {no_filter_result['error']}")
                results.append({
                    "dataset": "7scenes",
                    "scene": scene_name,
                    "method": "no_filter",
                    "threshold": None,
                    "frames": len(image_paths),
                    "success": False,
                    "inference_time_ms": None,
                    "memory_mb": None,
                    "compression_ratio": None,
                    "error": no_filter_result["error"],
                })
            
            # ===== 有过滤推理 =====
            for threshold in thresholds:
                print(f"    2️⃣  过滤推理 (τ={threshold})...", end="", flush=True)
                filter_result = comparator._try_infer_with_filter(
                    vgg_input, image_paths, threshold
                )
                
                if filter_result["success"]:
                    print(
                        f" ✓ (保留帧: {filter_result['kept_frames']}/{len(image_paths)}, "
                        f"时间: {filter_result['inference_time']:.2f}ms, "
                        f"显存: {filter_result['memory']:.0f}MB)"
                    )
                    results.append({
                        "dataset": "7scenes",
                        "scene": scene_name,
                        "method": "filter",
                        "threshold": threshold,
                        "frames": len(image_paths),
                        "kept_frames": filter_result["kept_frames"],
                        "success": True,
                        "inference_time_ms": filter_result["inference_time"],
                        "memory_mb": filter_result["memory"],
                        "compression_ratio": filter_result["compression_ratio"],
                        "error": None,
                    })
                else:
                    print(f" ❌ {filter_result['error']}")
                    results.append({
                        "dataset": "7scenes",
                        "scene": scene_name,
                        "method": "filter",
                        "threshold": threshold,
                        "frames": len(image_paths),
                        "kept_frames": None,
                        "success": False,
                        "inference_time_ms": None,
                        "memory_mb": None,
                        "compression_ratio": None,
                        "error": filter_result["error"],
                    })
        
        except Exception as e:
            print(f"    ❌ 场景处理失败: {str(e)[:80]}")
            
    return pd.DataFrame(results)


def evaluate_scannet(
    data_dir: str,
    num_scenes: Optional[int],
    thresholds: List[float],
    comparator: ReconstructionComparator,
    output_dir: str,
    input_frame: int,
) -> pd.DataFrame:
    """评估ScanNet数据集"""
    print(f"\n📊 开始评估 ScanNet 数据集...")
    
    # 获取场景列表
    if num_scenes is not None:
        scannet_scenes = get_all_scenes(data_dir, num_scenes)
    else:
        yaml_path = Path(__file__).parent / "scannet_50.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                scannet_scenes = [line.strip() for line in f if line.strip()]
        else:
            scannet_scenes = get_all_scenes(data_dir, 50)
    
    print(f"  找到 {len(scannet_scenes)} 个场景")
    
    results = []
    
    for scene_name in tqdm(scannet_scenes, desc="Scenes"):
        scene_dir = Path(data_dir) / scene_name
        
        if not (scene_dir / "color").exists():
            print(f"  ⚠️ 场景 {scene_name} 不存在，跳过")
            continue
        
        print(f"\n  📍 处理场景: {scene_name}")
        
        try:
            images_dir = scene_dir / "color"
            pose_path = scene_dir / "pose"
            
            image_paths = get_sorted_image_paths(images_dir)
            poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
            
            if poses_gt is None or len(image_paths) == 0:
                print(f"    ❌ 无效的场景数据")
                continue
            
            # 选择帧
            selected_frame_ids, selected_image_paths, selected_pose_indices = (
                build_frame_selection(image_paths, available_pose_frame_ids, input_frame)
            )
            
            if len(selected_image_paths) < 3:
                print(f"    ⚠️ 有效帧数不足")
                continue
            
            # 加载图像
            images = load_images_rgb(selected_image_paths)
            if not images:
                print(f"    ⚠️ 无法加载图像")
                continue
            
            images_array = np.stack(images)
            vgg_input, _, _ = get_vgg_input_imgs(images_array)  # 返回tensor
            
            # vgg_input已经是tensor [S, 3, H, W]，无需再转换
            vgg_input = vgg_input.float().to(comparator.device)
            
            print(f"    输入: {len(selected_image_paths)} 帧, 分辨率 {vgg_input.shape}")
            
            # ===== 无过滤推理 =====
            print(f"    1️⃣  无过滤推理...", end="", flush=True)
            no_filter_result = comparator._try_infer_no_filter(vgg_input, selected_image_paths)
            
            if no_filter_result["success"]:
                print(f" ✓ (时间: {no_filter_result['inference_time']:.2f}ms, 显存: {no_filter_result['memory']:.0f}MB)")
                results.append({
                    "dataset": "scannet",
                    "scene": scene_name,
                    "method": "no_filter",
                    "threshold": None,
                    "frames": len(selected_image_paths),
                    "success": True,
                    "inference_time_ms": no_filter_result["inference_time"],
                    "memory_mb": no_filter_result["memory"],
                    "compression_ratio": 1.0,
                    "error": None,
                })
            else:
                print(f" ❌ {no_filter_result['error']}")
                results.append({
                    "dataset": "scannet",
                    "scene": scene_name,
                    "method": "no_filter",
                    "threshold": None,
                    "frames": len(selected_image_paths),
                    "success": False,
                    "inference_time_ms": None,
                    "memory_mb": None,
                    "compression_ratio": None,
                    "error": no_filter_result["error"],
                })
            
            # ===== 有过滤推理 =====
            for threshold in thresholds:
                print(f"    2️⃣  过滤推理 (τ={threshold})...", end="", flush=True)
                filter_result = comparator._try_infer_with_filter(
                    vgg_input, selected_image_paths, threshold
                )
                
                if filter_result["success"]:
                    print(
                        f" ✓ (保留帧: {filter_result['kept_frames']}/{len(selected_image_paths)}, "
                        f"时间: {filter_result['inference_time']:.2f}ms, "
                        f"显存: {filter_result['memory']:.0f}MB)"
                    )
                    results.append({
                        "dataset": "scannet",
                        "scene": scene_name,
                        "method": "filter",
                        "threshold": threshold,
                        "frames": len(selected_image_paths),
                        "kept_frames": filter_result["kept_frames"],
                        "success": True,
                        "inference_time_ms": filter_result["inference_time"],
                        "memory_mb": filter_result["memory"],
                        "compression_ratio": filter_result["compression_ratio"],
                        "error": None,
                    })
                else:
                    print(f" ❌ {filter_result['error']}")
                    results.append({
                        "dataset": "scannet",
                        "scene": scene_name,
                        "method": "filter",
                        "threshold": threshold,
                        "frames": len(selected_image_paths),
                        "kept_frames": None,
                        "success": False,
                        "inference_time_ms": None,
                        "memory_mb": None,
                        "compression_ratio": None,
                        "error": filter_result["error"],
                    })
        
        except Exception as e:
            print(f"    ❌ 场景处理失败: {str(e)[:80]}")
    
    return pd.DataFrame(results)


def main(args):
    """主函数"""
    print("\n" + "="*80)
    print("过滤前后重建精度对比评估")
    print("="*80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析阈值
    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    print(f"\n📋 配置:")
    print(f"   数据集: {args.dataset_type}")
    print(f"   输出目录: {output_dir}")
    print(f"   过滤阈值: {thresholds}")
    
    # 加载模型
    print(f"\n🔄 加载模型...")
    device = args.device
    dtype = torch.float16
    
    model = VGGT(
        merging=args.merging,
        merge_ratio=args.merge_ratio,
    )
    
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    model = model.to(dtype)
    
    print(f"   ✓ 模型加载完成 (设备: {device}, dtype: {dtype})")
    
    # 创建对比器
    comparator = ReconstructionComparator(model, model.aggregator, device, dtype)
    
    # 根据数据集类型进行评估
    if args.dataset_type == "7scenes":
        scene_names = (
            args.scene_names.split(",")
            if args.scene_names
            else ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
        )
        results_df = evaluate_7scenes(
            args.data_dir,
            scene_names,
            thresholds,
            comparator,
            str(output_dir),
            args.input_frame,
        )
    else:  # scannet
        results_df = evaluate_scannet(
            args.data_dir,
            args.num_scenes,
            thresholds,
            comparator,
            str(output_dir),
            args.input_frame,
        )
    
    # 保存结果
    print(f"\n💾 保存结果...")
    csv_path = output_dir / "reconstruction_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV 已保存: {csv_path}")
    
    # 打印摘要
    print(f"\n📊 评估摘要:")
    print(f"   总行数: {len(results_df)}")
    
    if len(results_df) == 0:
        print("   ⚠️  无结果数据")
        return
    
    print(f"   成功: {results_df['success'].sum()}")
    print(f"   失败: {(~results_df['success']).sum()}")
    
    # 按方法统计
    print(f"\n   按方法分类:")
    for method in results_df['method'].unique():
        subset = results_df[results_df['method'] == method]
        success_count = subset['success'].sum()
        total_count = len(subset)
        print(f"     {method:12s}: {success_count:3d}/{total_count:3d} 成功")
    
    # 按OOM/错误统计
    print(f"\n   错误分类:")
    error_counts = results_df[~results_df['success']]['error'].value_counts()
    for error_type, count in error_counts.items():
        print(f"     {error_type:20s}: {count:3d}")
    
    # 性能对比（仅成功案例）
    print(f"\n⚡ 性能统计（仅成功案例）:")
    
    no_filter_data = results_df[(results_df['method'] == 'no_filter') & (results_df['success'])]
    if len(no_filter_data) > 0:
        avg_time_no_filter = no_filter_data['inference_time_ms'].mean()
        avg_memory_no_filter = no_filter_data['memory_mb'].mean()
        print(f"   无过滤:")
        print(f"     平均推理时间: {avg_time_no_filter:.2f} ms")
        print(f"     平均显存占用: {avg_memory_no_filter:.0f} MB")
    
    filter_data = results_df[(results_df['method'] == 'filter') & (results_df['success'])]
    if len(filter_data) > 0:
        avg_time_filter = filter_data['inference_time_ms'].mean()
        avg_memory_filter = filter_data['memory_mb'].mean()
        avg_compression = filter_data['compression_ratio'].mean()
        avg_kept_frames = filter_data['kept_frames'].mean()
        
        print(f"   有过滤 (平均):")
        print(f"     平均推理时间: {avg_time_filter:.2f} ms")
        print(f"     平均显存占用: {avg_memory_filter:.0f} MB")
        print(f"     平均压缩率: {avg_compression:.1%}")
        print(f"     平均保留帧数: {avg_kept_frames:.1f}")
        
        if len(no_filter_data) > 0:
            time_speedup = avg_time_no_filter / avg_time_filter
            memory_reduction = (avg_memory_no_filter - avg_memory_filter) / avg_memory_no_filter
            print(f"\n   相对改进:")
            print(f"     推理加速: {time_speedup:.2f}x")
            print(f"     显存降低: {memory_reduction:.1%}")
    
    print("\n" + "="*80)
    print("✅ 评估完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
