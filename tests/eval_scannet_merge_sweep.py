import argparse
import csv
import gc
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    build_frame_selection,
    evaluate_scene_and_save,
    get_sorted_image_paths,
    get_vgg_input_imgs,
    infer_vggt_and_reconstruct,
    load_images_rgb,
    load_poses,
)


METRIC_NAMES = [
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FastVGGT accuracy on ScanNet over sequence length and merge ratio"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/hba/Documents/Dataset/ScanNet/scans"),
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default=Path("/home/hba/Documents/Dataset/ScanNet/scans"),
    )
    parser.add_argument(
        "--scene_yaml",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/eval/scannet_50.yaml"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/tests/tests_result/scannet_merge_sweep"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
    )
    parser.add_argument(
        "--frame_counts",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100],
    )
    parser.add_argument(
        "--merge_ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.6, 0.9],
    )
    parser.add_argument("--merging", type=int, default=0)
    parser.add_argument("--num_scenes_per_config", type=int, default=2)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--depth_conf_thresh", type=float, default=1.0)
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16"])
    return parser.parse_args()


def load_scene_pool(scene_yaml: Path) -> List[str]:
    if not scene_yaml.exists():
        raise FileNotFoundError(f"Scene yaml not found: {scene_yaml}")
    with open(scene_yaml, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sample_scenes(scene_pool: List[str], num_scenes: int, seed: int) -> List[str]:
    if num_scenes <= 0:
        raise ValueError("num_scenes must be positive")
    if len(scene_pool) <= num_scenes:
        return list(scene_pool)
    rng = random.Random(seed)
    return sorted(rng.sample(scene_pool, num_scenes))


def append_row(csv_path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def merge_ratio_tag(merge_ratio: float) -> str:
    return str(merge_ratio).replace(".", "p")


def configure_model(model: VGGT, merging: int, merge_ratio: float) -> None:
    effective_merging = 25 if merge_ratio <= 0.0 else merging
    model.aggregator.merging = effective_merging
    model.aggregator.merge_ratio = merge_ratio
    for block in model.aggregator.frame_blocks:
        if hasattr(block, "attn"):
            block.attn.merge_ratio = merge_ratio
    for block in model.aggregator.global_blocks:
        if hasattr(block, "attn"):
            block.attn.merge_ratio = merge_ratio


def build_model(ckpt_path: str) -> VGGT:
    model = VGGT(merging=0, merge_ratio=0.0, vis_attn_map=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval().to(torch.float16)
    return model


def extract_metric_row(metrics: Optional[Dict[str, float]]) -> Dict[str, object]:
    row: Dict[str, object] = {metric: np.nan for metric in METRIC_NAMES}
    if metrics is None:
        return row
    for metric in METRIC_NAMES:
        if metric in metrics:
            row[metric] = float(metrics[metric])
    return row


def update_summary_from_detail(detail_csv: Path, summary_csv: Path) -> None:
    if not detail_csv.exists():
        return
    detail_df = pd.read_csv(detail_csv)
    if detail_df.empty:
        return

    summary_rows: List[Dict[str, object]] = []
    grouped = detail_df.groupby(["input_frame", "merge_ratio", "merging"], sort=True)
    for (input_frame, merge_ratio, merging), group in grouped:
        success_group = group[group["success"] == True].copy()  # noqa: E712
        row: Dict[str, object] = {
            "input_frame": int(input_frame),
            "merge_ratio": float(merge_ratio),
            "merging": int(merging),
            "success_scenes": int(len(success_group)),
            "total_scenes": int(len(group)),
            "sampled_scenes": "|".join(group["scene"].astype(str).tolist()),
        }
        for metric in METRIC_NAMES:
            row[metric] = (
                float(success_group[metric].mean()) if len(success_group) > 0 else np.nan
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["input_frame", "merge_ratio"]).reset_index(drop=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


def evaluate_single_scene(
    model: VGGT,
    scene: str,
    frame_count: int,
    merge_ratio: float,
    args: argparse.Namespace,
    output_scene_dir: Path,
) -> Dict[str, object]:
    scene_dir = args.data_dir / scene
    images_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"

    image_paths = get_sorted_image_paths(images_dir)
    poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_dir)
    if poses_gt is None or first_gt_pose is None or available_pose_frame_ids is None:
        raise RuntimeError("missing pose data")

    selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
        image_paths, available_pose_frame_ids, frame_count
    )
    c2ws = poses_gt[selected_pose_indices]

    if len(selected_image_paths) == 0:
        raise RuntimeError("no selected images")

    images = load_images_rgb(selected_image_paths)
    if not images or len(images) < 3:
        raise RuntimeError("insufficient valid images")

    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    model.update_patch_dimensions(patch_width, patch_height)

    (
        _extrinsic_np,
        _intrinsic_np,
        all_world_points,
        _all_point_colors,
        all_cam_to_world_mat,
        inference_time_ms,
    ) = infer_vggt_and_reconstruct(
        model,
        vgg_input,
        torch.float16,
        args.depth_conf_thresh,
        selected_image_paths,
    )

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

    row = {
        "scene": scene,
        "selected_frame_count": len(selected_image_paths),
    }
    row.update(extract_metric_row(metrics))
    return row


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(f"ckpt not found: {args.ckpt_path}")

    scene_pool = load_scene_pool(args.scene_yaml)
    sampled_scenes = sample_scenes(scene_pool, args.num_scenes_per_config, args.seed)

    output_root = args.output_path
    output_root.mkdir(parents=True, exist_ok=True)
    detail_csv = output_root / "scannet_merge_sweep_detail.csv"
    summary_csv = output_root / "scannet_merge_sweep_summary.csv"
    sampled_scene_txt = output_root / "sampled_scenes.txt"
    sampled_scene_txt.write_text("\n".join(sampled_scenes) + "\n", encoding="utf-8")

    if detail_csv.exists():
        detail_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()

    model = build_model(args.ckpt_path)

    detail_fields = [
        "scene",
        "input_frame",
        "merge_ratio",
        "merging",
        "selected_frame_count",
        "success",
        "error_msg",
        "chamfer_distance",
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "inference_time_ms",
    ]

    for frame_count in args.frame_counts:
        for merge_ratio in args.merge_ratios:
            print(f"\n=== Config: input_frame={frame_count}, merge_ratio={merge_ratio} ===")
            configure_model(model, args.merging, merge_ratio)

            for scene in sampled_scenes:
                output_scene_dir = (
                    output_root
                    / "scene_outputs"
                    / f"input_frame_{frame_count}"
                    / f"merge_ratio_{merge_ratio_tag(merge_ratio)}"
                    / scene
                )
                try:
                    result = evaluate_single_scene(
                        model=model,
                        scene=scene,
                        frame_count=frame_count,
                        merge_ratio=merge_ratio,
                        args=args,
                        output_scene_dir=output_scene_dir,
                    )
                    row = {
                        "scene": scene,
                        "input_frame": frame_count,
                        "merge_ratio": merge_ratio,
                        "merging": args.merging,
                        "selected_frame_count": result["selected_frame_count"],
                        "success": True,
                        "error_msg": "",
                        "chamfer_distance": result["chamfer_distance"],
                        "ate": result["ate"],
                        "are": result["are"],
                        "rpe_rot": result["rpe_rot"],
                        "rpe_trans": result["rpe_trans"],
                        "inference_time_ms": result["inference_time_ms"],
                    }
                    append_row(detail_csv, row, detail_fields)
                    print(
                        f"  ✓ {scene}: cd={row['chamfer_distance']:.4f}, ate={row['ate']:.4f}, time={row['inference_time_ms']:.2f}ms"
                    )
                except Exception as error:
                    row = {
                        "scene": scene,
                        "input_frame": frame_count,
                        "merge_ratio": merge_ratio,
                        "merging": args.merging,
                        "selected_frame_count": np.nan,
                        "success": False,
                        "error_msg": str(error),
                        "chamfer_distance": np.nan,
                        "ate": np.nan,
                        "are": np.nan,
                        "rpe_rot": np.nan,
                        "rpe_trans": np.nan,
                        "inference_time_ms": np.nan,
                    }
                    append_row(detail_csv, row, detail_fields)
                    print(f"  ✗ {scene}: {error}")
                    traceback.print_exc()
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            update_summary_from_detail(detail_csv, summary_csv)

    print("\nSaved files:")
    print(f"- Detail CSV : {detail_csv}")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Scenes TXT : {sampled_scene_txt}")


if __name__ == "__main__":
    main()
