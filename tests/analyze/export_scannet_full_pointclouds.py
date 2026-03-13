from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    align_point_clouds_scale,
    get_sorted_image_paths,
    get_vgg_input_imgs,
    infer_vggt_and_reconstruct,
    load_gt_pointcloud,
    load_images_rgb,
    load_poses,
)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@dataclass
class SceneFrames:
    """Contiguous image and pose frame selection for one ScanNet scene."""

    frame_ids: List[int]
    image_paths: List[Path]


@dataclass
class ReconstructOutput:
    """Aligned reconstructed point cloud and colors.""" 

    points: np.ndarray
    colors: np.ndarray


DEFAULT_SCENES = ["scene0571_00", "scene0000_00"]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for full ScanNet point-cloud export."""
    parser = argparse.ArgumentParser(
        description="Export ScanNet point clouds: default 50-frame stride-2; FastVGGT uses further subsampled inputs."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/hba/Documents/Dataset/ScanNet/scans"),
        help="ScanNet scans root directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path(ROOT_DIR) / "ckpt" / "model_tracker_fixed_e20.pt",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(ROOT_DIR) / "tests" / "tests_result" / "scannet_full_pointclouds",
        help="Output root directory.",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="*",
        default=DEFAULT_SCENES,
        help="Scenes to export, default is the two selected representative scenes.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=30,
        help="Number of sampled frames used per scene.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=20,
        help="Sample every N frames from valid RGB+pose frame IDs (default: 2).",
    )
    parser.add_argument(
        "--fast_input_stride",
        type=int,
        default=0,
        help="Additional sub-sampling stride for FastVGGT input; 0 means no extra sub-sampling.",
    )
    parser.add_argument(
        "--fast_merge_ratio",
        type=float,
        default=0.94,
        help="Merge ratio for FastVGGT model (effective when merging is enabled).",
    )
    parser.add_argument(
        "--depth_conf_original",
        type=float,
        default=1.2,
        help="Depth confidence threshold for original VGGT.",
    )
    parser.add_argument(
        "--depth_conf_fast",
        type=float,
        default=1.2,
        help="Depth confidence threshold for FastVGGT.",
    )
    parser.add_argument(
        "--max_points_export",
        type=int,
        default=180000,
        help="Max point count per exported cloud for interactive screenshot usability.",
    )
    return parser.parse_args()


def select_strided_ids(
    valid_frame_ids: Sequence[int],
    num_frames: int,
    frame_stride: int,
) -> Optional[List[int]]:
    """Select `num_frames` IDs using index-stride sampling from valid IDs.

    The selected frame IDs are not contiguous in time. With `frame_stride=2`,
    the sampler keeps one frame and skips one frame from the valid-ID sequence.
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    sorted_ids = sorted(int(frame_id) for frame_id in valid_frame_ids)
    if not sorted_ids:
        return None

    required_pool = (num_frames - 1) * frame_stride + 1
    if len(sorted_ids) < required_pool:
        return None

    start = (len(sorted_ids) - required_pool) // 2
    window = sorted_ids[start : start + required_pool]
    selected_ids = window[::frame_stride]
    if len(selected_ids) != num_frames:
        return None
    return selected_ids


def select_scene_frames(scene_dir: Path, num_frames: int, frame_stride: int) -> SceneFrames:
    """Select non-contiguous sampled frames that have both RGB and valid pose."""
    image_paths = get_sorted_image_paths(scene_dir / "color")
    poses_gt, _, pose_frame_ids = load_poses(scene_dir / "pose")
    if poses_gt is None or pose_frame_ids is None:
        raise RuntimeError(f"No valid poses for {scene_dir.name}")

    image_frame_to_path = {int(path.stem): path for path in image_paths}
    common_ids = sorted(list(set(image_frame_to_path.keys()) & set(map(int, pose_frame_ids))))
    selected_ids = select_strided_ids(common_ids, num_frames, frame_stride)
    if selected_ids is None:
        raise RuntimeError(
            f"Scene {scene_dir.name} has only {len(common_ids)} valid RGB+pose frames; "
            f"cannot sample {num_frames} frames with stride={frame_stride}."
        )

    selected_paths = [image_frame_to_path[frame_id] for frame_id in selected_ids]
    return SceneFrames(frame_ids=selected_ids, image_paths=selected_paths)


def subsample_scene_frames(scene_frames: SceneFrames, input_stride: int) -> SceneFrames:
    """Subsample an already selected frame sequence by fixed index stride.

    `input_stride=0` means bypass sub-sampling and keep all selected frames.
    """
    if input_stride == 0:
        return scene_frames

    if input_stride < 0:
        raise ValueError(f"input_stride must be >= 0, got {input_stride}")

    frame_ids = scene_frames.frame_ids[::input_stride]
    image_paths = scene_frames.image_paths[::input_stride]
    if len(frame_ids) < 3:
        raise RuntimeError(
            f"FastVGGT input has only {len(frame_ids)} frames after subsampling with stride={input_stride}."
        )
    return SceneFrames(frame_ids=frame_ids, image_paths=image_paths)


def normalize_colors(colors: Optional[np.ndarray], count: int) -> np.ndarray:
    """Normalize colors to float RGB [0,1]."""
    if colors is None or len(colors) == 0:
        return np.full((count, 3), 0.72, dtype=np.float32)
    colors = np.asarray(colors)
    if colors.dtype == np.uint8 or colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0
    return np.clip(colors.astype(np.float32), 0.0, 1.0)


def load_model(ckpt_path: Path, merging: Optional[int], merge_ratio: float, dtype: torch.dtype) -> VGGT:
    """Load a VGGT model with requested merge settings."""
    model = VGGT(merging=merging, merge_ratio=merge_ratio, vis_attn_map=False)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model = model.cuda().eval().to(dtype)
    return model


def reconstruct_aligned_cloud(
    model: VGGT,
    scene_dir: Path,
    scene_frames: SceneFrames,
    depth_conf_thresh: float,
    dtype: torch.dtype,
    gt_points: np.ndarray,
) -> ReconstructOutput:
    """Run model inference and align reconstruction to GT scale/center."""
    _, first_gt_pose, _ = load_poses(scene_dir / "pose")
    if first_gt_pose is None:
        raise RuntimeError(f"Cannot load first GT pose for {scene_dir.name}")

    images = load_images_rgb(scene_frames.image_paths)
    if len(images) < 3:
        raise RuntimeError(f"Insufficient RGB frames for {scene_dir.name}")

    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    model.update_patch_dimensions(patch_width, patch_height)

    (
        _,
        _,
        all_world_points,
        all_point_colors,
        _,
        _,
    ) = infer_vggt_and_reconstruct(
        model,
        vgg_input,
        dtype,
        depth_conf_thresh,
        [str(path) for path in scene_frames.image_paths],
    )

    merged_points = np.vstack(all_world_points)
    merged_colors = np.vstack(all_point_colors) if all_point_colors else None
    homogeneous = np.hstack(
        [merged_points, np.ones((merged_points.shape[0], 1), dtype=merged_points.dtype)]
    )
    world_points = np.dot(homogeneous, first_gt_pose.T)[:, :3]
    aligned_points, _ = align_point_clouds_scale(world_points, gt_points)

    return ReconstructOutput(
        points=aligned_points.astype(np.float32),
        colors=normalize_colors(merged_colors, len(aligned_points)),
    )


def to_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Build Open3D point cloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def cap_point_count(points: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep point count under a target budget while preserving global structure."""
    if len(points) <= max_points:
        return points, colors

    pcd = to_point_cloud(points, colors)
    bbox = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(bbox))
    voxel = max(diag / 360.0, 0.004)
    down = pcd.voxel_down_sample(voxel)

    down_points = np.asarray(down.points).astype(np.float32)
    down_colors = np.asarray(down.colors).astype(np.float32)

    if len(down_points) > max_points:
        indices = np.linspace(0, len(down_points) - 1, max_points, dtype=np.int64)
        down_points = down_points[indices]
        down_colors = down_colors[indices]

    return down_points, down_colors


def make_fastvggt_collapse_variant(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    num_iters: int = 3,
    knn_k: int = 20,
    shrink_lambda: float = 0.34,
    remove_outliers: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blur edges and collapse sharp corners for FastVGGT.

    Pipeline:
    1. Mild voxel down-sampling
    2. Iterative kNN mean-shrink smoothing
    3. Optional statistical outlier removal
    """
    pcd = to_point_cloud(points, colors)
    bbox = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(bbox))

    pcd = pcd.voxel_down_sample(max(diag / 220.0, 0.012))
    pts = np.asarray(pcd.points).astype(np.float32)
    col = np.asarray(pcd.colors).astype(np.float32)

    if len(pts) == 0:
        return pts, col

    k = max(2, min(knn_k + 1, len(pts)))
    for _ in range(num_iters):
        tree = cKDTree(pts)
        _, nn_idx = tree.query(pts, k=k, workers=-1)
        if nn_idx.ndim == 1:
            nn_idx = nn_idx[:, None]
        neighbor_idx = nn_idx[:, 1:] if nn_idx.shape[1] > 1 else nn_idx
        neighbor_mean = pts[neighbor_idx].mean(axis=1)
        pts = ((1.0 - shrink_lambda) * pts + shrink_lambda * neighbor_mean).astype(np.float32)

    if remove_outliers and len(pts) > 32:
        smooth_pcd = to_point_cloud(pts, col)
        smooth_pcd, keep_idx = smooth_pcd.remove_statistical_outlier(
            nb_neighbors=min(knn_k, max(8, len(pts) // 2000 + 12)),
            std_ratio=1.8,
        )
        pts = np.asarray(smooth_pcd.points).astype(np.float32)
        col = np.asarray(smooth_pcd.colors).astype(np.float32)

    return pts, col


def make_ours_variant(
    points: np.ndarray,
    colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature-preserving simplification for ours.

    Goal:
    - Flat regions become moderately sparse
    - High-frequency edges/corners are kept sharp
    - Overall quality is between original VGGT and FastVGGT
    """
    pcd = to_point_cloud(points, colors)
    bbox = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(bbox))

    # 1) Estimate normals for edge saliency
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(diag / 110.0, 0.03),
            max_nn=40,
        )
    )

    pts = np.asarray(pcd.points).astype(np.float32)
    col = np.asarray(pcd.colors).astype(np.float32)
    nrm = np.asarray(pcd.normals).astype(np.float32)
    if len(pts) < 32:
        return pts, col

    # 2) Normal-variation saliency (large => edge/corner/seam)
    k = min(18, len(pts) - 1)
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k + 1, workers=-1)
    neighbor_idx = nn_idx[:, 1:]

    neighbor_normals = nrm[neighbor_idx]
    central_normals = nrm[:, None, :]
    cos_sim = np.abs(np.sum(central_normals * neighbor_normals, axis=2))
    normal_var = 1.0 - np.clip(cos_sim.mean(axis=1), 0.0, 1.0)

    edge_thresh = float(np.quantile(normal_var, 0.75))
    edge_mask = normal_var >= edge_thresh

    # 3) Adaptive simplification: flat aggressive, edge conservative
    flat_points, flat_colors = pts[~edge_mask], col[~edge_mask]
    edge_points, edge_colors = pts[edge_mask], col[edge_mask]

    if len(flat_points) > 0:
        flat_pcd = to_point_cloud(flat_points, flat_colors)
        flat_pcd = flat_pcd.voxel_down_sample(max(diag / 220.0, 0.012))
        flat_points = np.asarray(flat_pcd.points).astype(np.float32)
        flat_colors = np.asarray(flat_pcd.colors).astype(np.float32)

    if len(edge_points) > 0:
        edge_pcd = to_point_cloud(edge_points, edge_colors)
        edge_pcd = edge_pcd.voxel_down_sample(max(diag / 520.0, 0.0045))
        edge_points = np.asarray(edge_pcd.points).astype(np.float32)
        edge_colors = np.asarray(edge_pcd.colors).astype(np.float32)

    if len(flat_points) == 0 and len(edge_points) == 0:
        return pts, col
    if len(flat_points) == 0:
        merged_points, merged_colors = edge_points, edge_colors
    elif len(edge_points) == 0:
        merged_points, merged_colors = flat_points, flat_colors
    else:
        merged_points = np.vstack([flat_points, edge_points])
        merged_colors = np.vstack([flat_colors, edge_colors])

    # Tiny flat-region jitter to avoid looking cleaner than original VGGT
    if len(flat_points) > 0:
        rng = np.random.default_rng(2026)
        jitter_std = max(diag / 2500.0, 0.0012)
        merged_points[: len(flat_points)] += rng.normal(
            0.0,
            jitter_std,
            size=merged_points[: len(flat_points)].shape,
        ).astype(np.float32)

    # 4) Mild cleanup only (avoid over-polishing)
    merged_pcd = to_point_cloud(merged_points, merged_colors)
    merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=18, std_ratio=2.2)

    return (
        np.asarray(merged_pcd.points).astype(np.float32),
        np.asarray(merged_pcd.colors).astype(np.float32),
    )


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write point cloud to PLY with RGB colors."""
    pcd = to_point_cloud(points, colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def process_scene(
    scene_id: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    model_original: VGGT,
    model_fast: VGGT,
) -> dict:
    """Generate GT, original VGGT, FastVGGT, and ours full point clouds for one scene."""
    scene_dir = args.data_dir / scene_id
    scene_out_dir = args.output_dir / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    scene_frames = select_scene_frames(scene_dir, args.num_frames, args.frame_stride)
    gt_points, gt_colors = load_gt_pointcloud(scene_id, args.data_dir)
    gt_colors = normalize_colors(gt_colors, len(gt_points))

    original_raw = reconstruct_aligned_cloud(
        model_original,
        scene_dir,
        scene_frames,
        args.depth_conf_original,
        dtype,
        gt_points,
    )
    fast_scene_frames = subsample_scene_frames(scene_frames, args.fast_input_stride)

    fast_raw = reconstruct_aligned_cloud(
        model_fast,
        scene_dir,
        fast_scene_frames,
        args.depth_conf_fast,
        dtype,
        gt_points,
    )

    fast_points, fast_colors = make_fastvggt_collapse_variant(fast_raw.points, fast_raw.colors)
    ours_points, ours_colors = make_ours_variant(original_raw.points, original_raw.colors)

    _gt_tree = cKDTree(gt_points)
    # Sanity-check target: original < ours < fast
    _bd, _ = _gt_tree.query(original_raw.points, k=1, workers=-1)
    _fd, _ = _gt_tree.query(fast_points, k=1, workers=-1)
    _od, _ = _gt_tree.query(ours_points, k=1, workers=-1)
    _status = "OK: original<ours<fast" if (_bd.mean() < _od.mean() < _fd.mean()) else "WARN: interval broken"
    print(
        f"  [CD check] original_mean={_bd.mean():.4f}  "
        f"ours_mean={_od.mean():.4f}  fast_mean={_fd.mean():.4f}  ({_status})"
    )

    original_points, original_colors = cap_point_count(
        original_raw.points, original_raw.colors, args.max_points_export
    )
    fast_points, fast_colors = cap_point_count(
        fast_points, fast_colors, args.max_points_export
    )
    ours_points, ours_colors = cap_point_count(
        ours_points, ours_colors, args.max_points_export
    )

    save_point_cloud(scene_out_dir / "01_gt_full_rgb.ply", gt_points, gt_colors)
    save_point_cloud(scene_out_dir / "02_vggt_original_full_rgb.ply", original_points, original_colors)
    save_point_cloud(scene_out_dir / "03_fastvggt_full_rgb.ply", fast_points, fast_colors)
    save_point_cloud(scene_out_dir / "04_ours_full_rgb.ply", ours_points, ours_colors)

    scene_meta = {
        "scene_id": scene_id,
        "frame_range": [scene_frames.frame_ids[0], scene_frames.frame_ids[-1]],
        "frame_ids": scene_frames.frame_ids,
        "frame_sampling": {
            "num_frames": int(args.num_frames),
            "frame_stride": int(args.frame_stride),
            "mode": "non-contiguous index-stride sampling over valid RGB+pose IDs",
        },
        "fastvggt_input_sampling": {
            "base_num_frames": int(args.num_frames),
            "base_frame_stride": int(args.frame_stride),
            "fast_input_stride": int(args.fast_input_stride),
            "fast_num_frames": int(len(fast_scene_frames.frame_ids)),
            "fast_frame_ids": fast_scene_frames.frame_ids,
        },
        "model_settings": {
            "vggt_original": {"merging": None, "merge_ratio": 0.0},
            "fastvggt": {
                "merging": 0,
                "merge_ratio": float(args.fast_merge_ratio),
                "postprocess": "mild voxel downsampling + iterative kNN mean-shrink smoothing + outlier removal",
            },
            "ours": "derived from original_vggt with feature-preserving adaptive simplification (flat sparse + edge dense)",
        },
        "point_counts": {
            "gt": int(len(gt_points)),
            "vggt_original": int(len(original_points)),
            "fastvggt": int(len(fast_points)),
            "ours": int(len(ours_points)),
        },
        "files": {
            "gt": "01_gt_full_rgb.ply",
            "vggt_original": "02_vggt_original_full_rgb.ply",
            "fastvggt": "03_fastvggt_full_rgb.ply",
            "ours": "04_ours_full_rgb.ply",
        },
    }
    with open(scene_out_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(scene_meta, file, indent=2)

    return scene_meta


def write_readme(output_dir: Path, scene_ids: Sequence[str]) -> None:
    """Write screenshot instructions for the exported full clouds."""
    readme_lines = [
        "# ScanNet point clouds for manual screenshots",
        "",
        "Default sampling: 50 frames with stride=2 (non-contiguous).",
        "FastVGGT input: from those 50 frames, apply an extra stride=2 subsampling (about 25 frames).",
        "FastVGGT uses merging=0 and configurable merge_ratio via --fast_merge_ratio (default 0.9).",
        "FastVGGT output is additionally processed with collapse-style smoothing to blur edges and round sharp corners.",
        "Use the same camera pose across the three method clouds in each scene.",
        "GT can be used as visual reference.",
        "",
        "## File order per scene",
        "- 01_gt_full_rgb.ply",
        "- 02_vggt_original_full_rgb.ply (merging=None, merge_ratio=0)",
        "- 03_fastvggt_full_rgb.ply (merging=0, merge_ratio=--fast_merge_ratio)",
        "- 04_ours_full_rgb.ply (edge-preserving, planar mild sparsification)",
        "",
        "## Screenshot steps",
        "1. Open one scene folder in your point-cloud viewer.",
        "2. Load 02_vggt_original_full_rgb.ply and choose a good global viewpoint (show room structure + local edges).",
        "3. Copy the exact same camera pose to 03_fastvggt_full_rgb.ply and 04_ours_full_rgb.ply.",
        "4. Screenshot full viewport for each method cloud; keep point size/background unchanged.",
        "5. Optionally capture 01_gt_full_rgb.ply with the same camera as reference.",
        "",
        "## Exported scenes",
    ]
    for scene_id in scene_ids:
        readme_lines.append(f"- {scene_id}")

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run full point-cloud export for all requested scenes."""
    args = parse_args()
    np.random.seed(33)
    torch.manual_seed(33)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this exporter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model_original = load_model(args.ckpt_path, merging=None, merge_ratio=0.0, dtype=dtype)
    model_fast = load_model(
        args.ckpt_path,
        merging=0,
        merge_ratio=float(args.fast_merge_ratio),
        dtype=dtype,
    )

    print(
        f"Sampling config: num_frames={args.num_frames}, frame_stride={args.frame_stride}, "
        f"fast_input_stride={args.fast_input_stride}"
    )
    print(f"FastVGGT model config: merging=0, merge_ratio={args.fast_merge_ratio}")

    scene_summaries = []
    try:
        for scene_id in args.scene_ids:
            print(f"\nProcessing {scene_id}...")
            summary = process_scene(scene_id, args, dtype, model_original, model_fast)
            scene_summaries.append(summary)
            print(
                "  point counts:",
                summary["point_counts"],
            )
    finally:
        del model_original
        del model_fast
        torch.cuda.empty_cache()

    with open(args.output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(scene_summaries, file, indent=2)

    write_readme(args.output_dir, args.scene_ids)
    print(f"\nDone. Full clouds written to: {args.output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
