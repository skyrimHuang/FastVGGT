from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import get_vgg_input_imgs, infer_vggt_and_reconstruct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct dense point cloud from a video using original VGGT (no token merging)."
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Input video path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/demo_output/video_reconstruction_original_vggt"),
        help="Output directory for PLY and metadata.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt"),
        help="VGGT checkpoint path.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum number of frames sampled from the video.",
    )
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=2000000,
        help="Maximum number of output points kept in the final point cloud.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Random seed for reproducible downsampling.",
    )
    return parser.parse_args()


def sample_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    keep = min(total_frames, max_frames)
    return np.linspace(0, total_frames - 1, num=keep, dtype=np.int64)


def load_video_frames(video_path: Path, max_frames: int) -> Tuple[List[np.ndarray], List[str], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_indices = sample_frame_indices(total_frames, max_frames)
    target_set = set(frame_indices.tolist()) if len(frame_indices) > 0 else None

    frames_rgb: List[np.ndarray] = []
    frame_names: List[str] = []

    frame_id = 0
    selected_id = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        use_frame = False
        if target_set is None:
            use_frame = True
        else:
            if frame_id in target_set:
                use_frame = True

        if use_frame:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            frame_names.append(f"frame_{frame_id:06d}.jpg")
            selected_id += 1
            if target_set is None and selected_id >= max_frames:
                break

        frame_id += 1

    cap.release()

    if len(frames_rgb) == 0:
        raise RuntimeError("No frames were extracted from video.")

    return frames_rgb, frame_names, total_frames


def load_model(ckpt_path: Path, dtype: torch.dtype) -> torch.nn.Module:
    model = VGGT(merging=None, merge_ratio=0.0, vis_attn_map=False)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(dtype)
    return model


def merge_and_limit_points(
    all_world_points: Sequence[np.ndarray],
    all_point_colors: Sequence[np.ndarray],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.vstack(all_world_points).astype(np.float32)
    colors = np.vstack(all_point_colors).astype(np.uint8)

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]

    if len(points) > max_points:
        random.seed(seed)
        np.random.seed(seed)
        idx = np.random.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write PLY: {path}")


def scale_intrinsics_to_original_resolution(
    intrinsic_np: np.ndarray,
    input_width: int,
    input_height: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    if intrinsic_np.ndim != 3 or intrinsic_np.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape [N,3,3], got {intrinsic_np.shape}")

    scale_x = float(original_width) / float(input_width)
    scale_y = float(original_height) / float(input_height)
    intrinsic_scaled = intrinsic_np.copy().astype(np.float32)
    intrinsic_scaled[:, 0, 0] *= scale_x
    intrinsic_scaled[:, 1, 1] *= scale_y
    intrinsic_scaled[:, 0, 2] *= scale_x
    intrinsic_scaled[:, 1, 2] *= scale_y
    return intrinsic_scaled


def save_camera_parameters(
    output_dir: Path,
    frame_names: Sequence[str],
    extrinsic_np: np.ndarray,
    intrinsic_np: np.ndarray,
    intrinsic_scaled_np: np.ndarray,
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> dict:
    camera_dir = output_dir / "camera_params"
    camera_dir.mkdir(parents=True, exist_ok=True)

    np.save(camera_dir / "extrinsics_w2c.npy", extrinsic_np.astype(np.float32))
    np.save(camera_dir / "intrinsics_input_res.npy", intrinsic_np.astype(np.float32))
    np.save(camera_dir / "intrinsics_original_res.npy", intrinsic_scaled_np.astype(np.float32))

    num_frames = int(extrinsic_np.shape[0])
    c2w_list: List[np.ndarray] = []
    for idx in range(num_frames):
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :4] = extrinsic_np[idx]
        c2w_4x4 = np.linalg.inv(w2c_4x4).astype(np.float32)
        c2w_list.append(c2w_4x4)
    c2w_arr = np.stack(c2w_list, axis=0)
    np.save(camera_dir / "extrinsics_c2w.npy", c2w_arr)

    per_frame = []
    for idx in range(num_frames):
        frame_name = frame_names[idx] if idx < len(frame_names) else f"frame_{idx:06d}.jpg"
        per_frame.append(
            {
                "frame_index": idx,
                "frame_name": frame_name,
                "intrinsic_input_res_3x3": intrinsic_np[idx].astype(float).tolist(),
                "intrinsic_original_res_3x3": intrinsic_scaled_np[idx].astype(float).tolist(),
                "extrinsic_w2c_3x4": extrinsic_np[idx].astype(float).tolist(),
                "extrinsic_c2w_4x4": c2w_arr[idx].astype(float).tolist(),
            }
        )

    camera_json_path = camera_dir / "camera_params_per_frame.json"
    camera_json_path.write_text(
        json.dumps(
            {
                "convention": {
                    "w2c": "world_to_camera 3x4 from VGGT",
                    "c2w": "inverse transform of w2c",
                },
                "image_resolution": {
                    "input_to_vggt": [int(input_width), int(input_height)],
                    "original_video": [int(original_width), int(original_height)],
                },
                "num_frames": num_frames,
                "frames": per_frame,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "camera_dir": str(camera_dir),
        "intrinsics_input_res_npy": str(camera_dir / "intrinsics_input_res.npy"),
        "intrinsics_original_res_npy": str(camera_dir / "intrinsics_original_res.npy"),
        "extrinsics_w2c_npy": str(camera_dir / "extrinsics_w2c.npy"),
        "extrinsics_c2w_npy": str(camera_dir / "extrinsics_c2w.npy"),
        "per_frame_json": str(camera_json_path),
    }


def main() -> None:
    args = parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VGGT reconstruction, but no CUDA device is available.")

    compute_capability = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16

    print(f"[1/4] Loading video: {args.video_path}")
    frames_rgb, frame_names, total_frames = load_video_frames(args.video_path, args.max_frames)
    print(f"  - total video frames: {total_frames}")
    print(f"  - sampled frames: {len(frames_rgb)}")

    if len(frames_rgb) < 3:
        raise RuntimeError("At least 3 valid frames are required for reconstruction.")

    print("[2/4] Preparing VGGT input")
    images_array = np.stack(frames_rgb, axis=0)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    print(f"  - input tensor: {tuple(vgg_input.shape)}")
    print(f"  - patch dims: {patch_width}x{patch_height}")

    print("[3/4] Running original VGGT (merging=None, merge_ratio=0.0)")
    start_time = time.time()
    model = load_model(args.ckpt_path, dtype=dtype)
    model.update_patch_dimensions(patch_width, patch_height)
    (
        extrinsic_np,
        intrinsic_np,
        all_world_points,
        all_point_colors,
        _all_cam_to_world_mat,
        inference_time_ms,
    ) = infer_vggt_and_reconstruct(
        model=model,
        vgg_input=vgg_input,
        dtype=dtype,
        depth_conf_thresh=args.depth_conf_thresh,
        image_paths=frame_names,
    )

    print("[4/4] Merging and exporting dense point cloud")
    merged_points, merged_colors = merge_and_limit_points(
        all_world_points=all_world_points,
        all_point_colors=all_point_colors,
        max_points=args.max_points,
        seed=args.seed,
    )

    ply_path = args.output_dir / "reconstructed_points_original_vggt_dense.ply"
    save_ply(ply_path, merged_points, merged_colors)

    input_height, input_width = int(vgg_input.shape[2]), int(vgg_input.shape[3])
    original_height, original_width = int(frames_rgb[0].shape[0]), int(frames_rgb[0].shape[1])
    intrinsic_scaled_np = scale_intrinsics_to_original_resolution(
        intrinsic_np=intrinsic_np,
        input_width=input_width,
        input_height=input_height,
        original_width=original_width,
        original_height=original_height,
    )
    camera_output_paths = save_camera_parameters(
        output_dir=args.output_dir,
        frame_names=frame_names,
        extrinsic_np=extrinsic_np,
        intrinsic_np=intrinsic_np,
        intrinsic_scaled_np=intrinsic_scaled_np,
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
    )

    total_time_s = time.time() - start_time
    meta = {
        "video_path": str(args.video_path),
        "model": {
            "type": "VGGT",
            "merging": None,
            "merge_ratio": 0.0,
            "ckpt_path": str(args.ckpt_path),
        },
        "reconstruction": {
            "sampled_frames": len(frames_rgb),
            "depth_conf_thresh": args.depth_conf_thresh,
            "inference_time_ms": float(inference_time_ms),
            "final_point_count": int(len(merged_points)),
            "max_points": int(args.max_points),
            "dtype": str(dtype),
            "input_resolution": [input_width, input_height],
            "original_video_resolution": [original_width, original_height],
            "total_runtime_s": float(total_time_s),
        },
        "outputs": {
            "point_cloud_ply": str(ply_path),
            "camera_parameters": camera_output_paths,
        },
    }
    meta_path = args.output_dir / "reconstruction_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ Reconstruction finished")
    print(f"📦 PLY: {ply_path}")
    print(f"🧾 Metadata: {meta_path}")


if __name__ == "__main__":
    main()
