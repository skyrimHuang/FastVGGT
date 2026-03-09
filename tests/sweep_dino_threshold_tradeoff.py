"""
DINOv2关键帧阈值扫参与折中点推荐实验。

目标：找到在精度变化较小前提下，显著降低推理时间的阈值 tau。

输出：
  1) dino_threshold_sweep.csv：每个tau的性能/精度与基线对比
  2) recommendation.json：自动推荐阈值与排序结果

示例：
  conda activate fastvggt && python tests/sweep_dino_threshold_tradeoff.py \
    --scene_names chess \
    --requested_frames 300 \
    --taus 0.001,0.0015,0.002,0.003,0.004,0.006 \
    --output_dir ./tests/dino_tau_sweep
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    get_vgg_input_imgs,
    load_images_rgb,
)

from tests.eval_7scenes_3strategies_long import (
    KeyframeStrategyEvaluator,
    compute_trajectory_metrics_safe,
    extract_poses_from_predictions,
    get_7scenes_sequences,
    load_poses,
    normalize_keyframe_indices,
)


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)


def _safe_delta(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base is None:
        return None
    return float(cur) - float(base)


def _parse_taus(taus_text: str) -> List[float]:
    values = []
    for token in taus_text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("--taus 不能为空")
    return sorted(set(values))


def _load_seq_data(color_dir: Path, pose_dir: Path, actual_frames: int, device: str):
    color_files = sorted(color_dir.glob("*.color.png"))[:actual_frames]
    image_paths = [str(f) for f in color_files]

    images = load_images_rgb(image_paths)
    if not images or len(images) < 3:
        raise RuntimeError("无法加载足够图像")

    images_array = np.stack(images)
    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    vgg_input = vgg_input.float().to(device)

    poses_gt, _, available_frame_ids = load_poses(pose_dir)
    if poses_gt is None or available_frame_ids is None or len(available_frame_ids) == 0:
        raise RuntimeError("无法加载位姿")

    available_frame_ids = np.array(available_frame_ids, dtype=np.int64)
    valid_frame_count = min(len(image_paths), len(available_frame_ids))
    if valid_frame_count < 3:
        raise RuntimeError("有效帧-位姿对不足")

    vgg_input = vgg_input[:valid_frame_count]
    image_paths = image_paths[:valid_frame_count]
    poses_gt = poses_gt[:valid_frame_count]
    available_frame_ids = available_frame_ids[:valid_frame_count]

    return vgg_input, image_paths, poses_gt, available_frame_ids, valid_frame_count


def _run_full_baseline(
    evaluator: KeyframeStrategyEvaluator,
    vgg_input: torch.Tensor,
    image_paths: List[str],
    poses_gt: np.ndarray,
    frame_ids: np.ndarray,
) -> Dict:
    result = evaluator._try_infer_full(vgg_input, image_paths)

    row = {
        "strategy": "A_Full",
        "tau": None,
        "success": result["success"],
        "kept_frames": result.get("kept_frames"),
        "compression_ratio": result.get("compression_ratio"),
        "total_time_s": result.get("time_s"),
        "memory_mb": result.get("memory_mb"),
        "ate": None,
        "are": None,
        "rpe_trans": None,
        "rpe_rot": None,
        "error": result.get("error"),
    }

    if result["success"]:
        poses_est = extract_poses_from_predictions(result.get("predictions"), vgg_input.shape)
        if poses_est is not None and len(poses_est) == len(poses_gt):
            metrics = compute_trajectory_metrics_safe(poses_est, poses_gt, frame_ids)
            if metrics is not None:
                row["ate"] = metrics.get("ate")
                row["are"] = metrics.get("are")
                row["rpe_trans"] = metrics.get("rpe_trans")
                row["rpe_rot"] = metrics.get("rpe_rot")

    return row


def _run_dino_one_tau(
    evaluator: KeyframeStrategyEvaluator,
    tau: float,
    vgg_input: torch.Tensor,
    image_paths: List[str],
    poses_gt: np.ndarray,
    frame_ids: np.ndarray,
) -> Dict:
    result = evaluator._try_infer_dino_filter_with_reuse(vgg_input, image_paths, threshold=tau)

    row = {
        "strategy": "C_DINOReuse",
        "tau": tau,
        "success": result["success"],
        "kept_frames": result.get("kept_frames"),
        "compression_ratio": result.get("compression_ratio"),
        "total_time_s": result.get("time_s"),
        "memory_mb": result.get("memory_mb"),
        "ate": None,
        "are": None,
        "rpe_trans": None,
        "rpe_rot": None,
        "error": result.get("error"),
    }

    if result["success"]:
        keyframe_indices = normalize_keyframe_indices(result.get("keyframe_indices"))
        poses_est = extract_poses_from_predictions(result.get("predictions"), vgg_input.shape)
        if (
            poses_est is not None
            and keyframe_indices is not None
            and len(poses_est) == len(keyframe_indices)
            and len(keyframe_indices) > 1
        ):
            poses_gt_k = poses_gt[keyframe_indices]
            frame_ids_k = frame_ids[keyframe_indices]
            metrics = compute_trajectory_metrics_safe(poses_est, poses_gt_k, frame_ids_k)
            if metrics is not None:
                row["ate"] = metrics.get("ate")
                row["are"] = metrics.get("are")
                row["rpe_trans"] = metrics.get("rpe_trans")
                row["rpe_rot"] = metrics.get("rpe_rot")

    return row


def _pick_recommendation(df: pd.DataFrame, args: argparse.Namespace) -> Dict:
    candidates = df[(df["strategy"] == "C_DINOReuse") & (df["success"] == True)].copy()
    if len(candidates) == 0:
        return {"status": "no_candidate", "reason": "所有DINO阈值都失败"}

    grouped = (
        candidates.groupby("tau", as_index=False)
        .agg(
            num_sequences=("scene", "count"),
            speedup_mean=("speedup_vs_full", "mean"),
            speedup_min=("speedup_vs_full", "min"),
            ate_ratio_mean=("ate_ratio_vs_full", "mean"),
            are_ratio_mean=("are_ratio_vs_full", "mean"),
            ate_ratio_max=("ate_ratio_vs_full", "max"),
            are_ratio_max=("are_ratio_vs_full", "max"),
            ate_delta_max=("ate_delta_vs_full", "max"),
            are_delta_max=("are_delta_vs_full", "max"),
            rpe_t_delta_max=("rpe_trans_delta_vs_full", "max"),
            rpe_r_delta_max=("rpe_rot_delta_vs_full", "max"),
            kept_ratio_mean=("kept_ratio_vs_full", "mean"),
        )
        .sort_values("tau")
    )

    strict = grouped[
        (grouped["speedup_min"] >= args.min_speedup)
        & (grouped["ate_ratio_max"] <= args.max_ate_ratio)
        & (grouped["are_ratio_max"] <= args.max_are_ratio)
        & (grouped["ate_delta_max"] <= args.max_ate_delta)
        & (grouped["are_delta_max"] <= args.max_are_delta)
        & (grouped["rpe_t_delta_max"] <= args.max_rpe_t_delta)
        & (grouped["rpe_r_delta_max"] <= args.max_rpe_r_delta)
    ].copy()

    criteria = {
        "min_speedup": args.min_speedup,
        "max_ate_ratio": args.max_ate_ratio,
        "max_are_ratio": args.max_are_ratio,
        "max_ate_delta": args.max_ate_delta,
        "max_are_delta": args.max_are_delta,
        "max_rpe_t_delta": args.max_rpe_t_delta,
        "max_rpe_r_delta": args.max_rpe_r_delta,
    }

    if len(strict) == 0:
        relaxed = grouped.sort_values(
            by=["ate_ratio_max", "are_ratio_max", "speedup_mean"],
            ascending=[True, True, False],
        )
        return {
            "status": "no_strict_match",
            "reason": "没有tau在所有序列上同时满足硬约束，返回最优折中",
            "criteria": criteria,
            "best_effort": relaxed.iloc[0].to_dict(),
            "all_tau_summary": grouped.to_dict(orient="records"),
        }

    strict = strict.sort_values(
        by=["speedup_mean", "ate_ratio_max", "are_ratio_max"],
        ascending=[False, True, True],
    )
    return {
        "status": "ok",
        "criteria": criteria,
        "recommended": strict.iloc[0].to_dict(),
        "top_candidates": strict.head(min(5, len(strict))).to_dict(orient="records"),
        "all_tau_summary": grouped.to_dict(orient="records"),
    }


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dino_threshold_sweep.csv"
    rec_path = output_dir / "recommendation.json"

    taus = _parse_taus(args.taus)
    print("=" * 80)
    print("DINO阈值扫参：速度-精度折中实验")
    print("=" * 80)
    print(f"场景: {args.scene_names}")
    print(f"requested_frames: {args.requested_frames}")
    print(f"taus: {taus}")
    print(f"输出目录: {output_dir}")

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
    evaluator = KeyframeStrategyEvaluator(model, device, dtype)

    scene_names = [s.strip() for s in args.scene_names.split(",") if s.strip()]
    sequences = get_7scenes_sequences(args.data_dir, scene_names, args.requested_frames)
    if len(sequences) == 0:
        raise RuntimeError("未找到有效序列")

    fieldnames = [
        "scene",
        "sequence",
        "requested_frames",
        "actual_frames",
        "available_frames",
        "strategy",
        "tau",
        "success",
        "kept_frames",
        "compression_ratio",
        "total_time_s",
        "fps",
        "memory_mb",
        "ate",
        "are",
        "rpe_trans",
        "rpe_rot",
        "ate_ratio_vs_full",
        "are_ratio_vs_full",
        "rpe_trans_ratio_vs_full",
        "rpe_rot_ratio_vs_full",
        "ate_delta_vs_full",
        "are_delta_vs_full",
        "rpe_trans_delta_vs_full",
        "rpe_rot_delta_vs_full",
        "speedup_vs_full",
        "time_ratio_vs_full",
        "kept_ratio_vs_full",
        "error",
    ]

    all_rows: List[Dict] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for idx, seq in enumerate(sequences):
            print(f"\n[{idx+1}/{len(sequences)}] {seq['scene']}/{seq['sequence']}")
            color_dir = Path(seq["color_dir"])
            pose_dir = Path(seq["pose_dir"])

            vgg_input, image_paths, poses_gt, frame_ids, valid_count = _load_seq_data(
                color_dir, pose_dir, seq["actual_frames"], device
            )
            print(f"  有效帧: {valid_count}")

            full = _run_full_baseline(evaluator, vgg_input, image_paths, poses_gt, frame_ids)
            base_time = full.get("total_time_s")
            base_kept = full.get("kept_frames")
            base_ate = full.get("ate")
            base_are = full.get("are")
            base_rpe_t = full.get("rpe_trans")
            base_rpe_r = full.get("rpe_rot")

            full_row = {
                "scene": seq["scene"],
                "sequence": seq["sequence"],
                "requested_frames": seq["requested_frames"],
                "actual_frames": valid_count,
                "available_frames": seq["available_frames"],
                "strategy": full["strategy"],
                "tau": None,
                "success": full["success"],
                "kept_frames": full.get("kept_frames"),
                "compression_ratio": full.get("compression_ratio"),
                "total_time_s": full.get("total_time_s"),
                "fps": _safe_ratio(valid_count, full.get("total_time_s")),
                "memory_mb": full.get("memory_mb"),
                "ate": full.get("ate"),
                "are": full.get("are"),
                "rpe_trans": full.get("rpe_trans"),
                "rpe_rot": full.get("rpe_rot"),
                "ate_ratio_vs_full": 1.0,
                "are_ratio_vs_full": 1.0,
                "rpe_trans_ratio_vs_full": 1.0,
                "rpe_rot_ratio_vs_full": 1.0,
                "ate_delta_vs_full": 0.0,
                "are_delta_vs_full": 0.0,
                "rpe_trans_delta_vs_full": 0.0,
                "rpe_rot_delta_vs_full": 0.0,
                "speedup_vs_full": 1.0,
                "time_ratio_vs_full": 1.0,
                "kept_ratio_vs_full": 1.0,
                "error": full.get("error"),
            }
            writer.writerow(full_row)
            all_rows.append(full_row)
            f.flush()

            if full["success"]:
                print(
                    f"  A_Full: {full['total_time_s']:.2f}s, "
                    f"ATE={full.get('ate')}, ARE={full.get('are')}"
                )
            else:
                print(f"  A_Full失败: {full.get('error')}")

            for tau in taus:
                dino = _run_dino_one_tau(
                    evaluator,
                    tau,
                    vgg_input,
                    image_paths,
                    poses_gt,
                    frame_ids,
                )

                row = {
                    "scene": seq["scene"],
                    "sequence": seq["sequence"],
                    "requested_frames": seq["requested_frames"],
                    "actual_frames": valid_count,
                    "available_frames": seq["available_frames"],
                    "strategy": dino["strategy"],
                    "tau": tau,
                    "success": dino["success"],
                    "kept_frames": dino.get("kept_frames"),
                    "compression_ratio": dino.get("compression_ratio"),
                    "total_time_s": dino.get("total_time_s"),
                    "fps": _safe_ratio(valid_count, dino.get("total_time_s")),
                    "memory_mb": dino.get("memory_mb"),
                    "ate": dino.get("ate"),
                    "are": dino.get("are"),
                    "rpe_trans": dino.get("rpe_trans"),
                    "rpe_rot": dino.get("rpe_rot"),
                    "ate_ratio_vs_full": _safe_ratio(dino.get("ate"), base_ate),
                    "are_ratio_vs_full": _safe_ratio(dino.get("are"), base_are),
                    "rpe_trans_ratio_vs_full": _safe_ratio(dino.get("rpe_trans"), base_rpe_t),
                    "rpe_rot_ratio_vs_full": _safe_ratio(dino.get("rpe_rot"), base_rpe_r),
                    "ate_delta_vs_full": _safe_delta(dino.get("ate"), base_ate),
                    "are_delta_vs_full": _safe_delta(dino.get("are"), base_are),
                    "rpe_trans_delta_vs_full": _safe_delta(dino.get("rpe_trans"), base_rpe_t),
                    "rpe_rot_delta_vs_full": _safe_delta(dino.get("rpe_rot"), base_rpe_r),
                    "speedup_vs_full": _safe_ratio(base_time, dino.get("total_time_s")),
                    "time_ratio_vs_full": _safe_ratio(dino.get("total_time_s"), base_time),
                    "kept_ratio_vs_full": _safe_ratio(dino.get("kept_frames"), base_kept),
                    "error": dino.get("error"),
                }

                writer.writerow(row)
                all_rows.append(row)
                f.flush()

                if dino["success"]:
                    time_text = (
                        f"{dino['total_time_s']:.2f}s"
                        if dino.get("total_time_s") is not None
                        else "None"
                    )
                    speedup_val = row.get("speedup_vs_full")
                    speedup_text = (
                        f"{speedup_val:.2f}x" if speedup_val is not None else "None"
                    )
                    print(
                        f"  tau={tau:<7g} -> t={time_text}, "
                        f"K={dino.get('kept_frames')}, ATE={dino.get('ate')}, "
                        f"speedup={speedup_text}"
                    )
                else:
                    print(f"  tau={tau:<7g} -> 失败: {dino.get('error')}")

            del vgg_input
            gc.collect()
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    rec = _pick_recommendation(df, args)
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"CSV: {csv_path}")
    print(f"推荐: {rec_path}")
    if rec.get("status") == "ok":
        best = rec["recommended"]
        print(
            f"推荐tau={best['tau']} | speedup均值={best['speedup_mean']:.2f}x "
            f"(最小={best['speedup_min']:.2f}x) | "
            f"ATE最差比={best['ate_ratio_max']:.3f} | ARE最差比={best['are_ratio_max']:.3f}"
        )
    elif rec.get("status") == "no_strict_match":
        best = rec.get("best_effort", {})
        if best:
            print(
                f"推荐状态: {rec.get('status')} ({rec.get('reason')}) | "
                f"best_effort tau={best.get('tau')} speedup均值={best.get('speedup_mean')}x"
            )
        else:
            print(f"推荐状态: {rec.get('status')} ({rec.get('reason')})")
    else:
        print(f"推荐状态: {rec.get('status')} ({rec.get('reason')})")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DINO threshold sweep")
    parser.add_argument(
        "--scene_names",
        type=str,
        default="chess",
        help="7Scenes场景，逗号分隔",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/hba/Documents/Dataset/7_scenes",
        help="7Scenes数据路径",
    )
    parser.add_argument(
        "--requested_frames",
        type=int,
        default=300,
        help="每序列请求帧数",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="0.001,0.0015,0.002,0.003,0.004,0.006",
        help="待扫描的DINO阈值列表（逗号分隔）",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tests/dino_tau_sweep",
        help="输出目录",
    )

    parser.add_argument(
        "--min_speedup",
        type=float,
        default=2.0,
        help="最低速度提升倍数约束",
    )
    parser.add_argument(
        "--max_ate_ratio",
        type=float,
        default=1.25,
        help="ATE相对基线最大倍率",
    )
    parser.add_argument(
        "--max_are_ratio",
        type=float,
        default=1.25,
        help="ARE相对基线最大倍率",
    )
    parser.add_argument(
        "--max_rpe_t_ratio",
        type=float,
        default=100.0,
        help="RPE平移相对基线最大倍率（默认放宽，建议看绝对增量约束）",
    )
    parser.add_argument(
        "--max_rpe_r_ratio",
        type=float,
        default=100.0,
        help="RPE旋转相对基线最大倍率（默认放宽，建议看绝对增量约束）",
    )
    parser.add_argument(
        "--max_ate_delta",
        type=float,
        default=0.2,
        help="ATE相对基线最大绝对增量",
    )
    parser.add_argument(
        "--max_are_delta",
        type=float,
        default=5.0,
        help="ARE相对基线最大绝对增量(度)",
    )
    parser.add_argument(
        "--max_rpe_t_delta",
        type=float,
        default=0.2,
        help="RPE平移相对基线最大绝对增量",
    )
    parser.add_argument(
        "--max_rpe_r_delta",
        type=float,
        default=5.0,
        help="RPE旋转相对基线最大绝对增量(度)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
