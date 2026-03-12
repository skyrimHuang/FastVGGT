#!/usr/bin/env python3
"""
Section 3.3.3 robustness validation on 7-Scenes (Office, RedKitchen).

Methods:
  A: FPFH coarse + point-to-point ICP
  B: DINOv2 semantic coarse only (RANSAC + Umeyama)
  C: DINOv2 semantic coarse + point-to-plane ICP

Metrics:
  - Recall: (rotation_error_deg < 5) and (translation_error_m < 0.05)
  - ICP convergence iterations (actual iterative steps)
  - Chamfer Distance (CD)

Outputs:
  tests/tests_result/hybrid_registration_7scenes/
    - hybrid_registration_pair_results.csv
    - hybrid_registration_summary.csv
"""

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import get_vgg_input_imgs, umeyama_alignment


@dataclass
class FrameData:
    scene: str
    seq: str
    frame_id: int
    color_path: Path
    depth_path: Path
    pose_path: Path
    pose_c2w: np.ndarray
    image_resized: np.ndarray
    depth_resized_m: np.ndarray
    intrinsics_resized: np.ndarray
    dino_tokens: np.ndarray
    global_desc: np.ndarray


@dataclass
class PairData:
    scene: str
    seq: str
    src: FrameData
    dst: FrameData
    src_submap: List[FrameData]
    dst_submap: List[FrameData]
    gt_t_dst_src: np.ndarray
    baseline_rot_deg: float
    baseline_trans_m: float


def read_test_sequences(scene_dir: Path) -> List[str]:
    test_split = scene_dir / "TestSplit.txt"
    with test_split.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    seqs = []
    for line in lines:
        num = "".join(ch for ch in line if ch.isdigit())
        seqs.append(f"seq-{int(num):02d}")
    return seqs


def load_pose_c2w(pose_path: Path) -> np.ndarray:
    pose = np.loadtxt(str(pose_path), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Invalid pose shape at {pose_path}: {pose.shape}")
    return pose


def resize_depth_to_model(depth_mm: np.ndarray, out_h: int = 392, out_w: int = 518) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[depth_m > 10.0] = 0.0
    depth_m[depth_m < 1e-4] = 0.0
    depth_resized = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return depth_resized


def resize_rgb_to_model(img_bgr: np.ndarray, out_h: int = 392, out_w: int = 518) -> np.ndarray:
    resized = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


def intrinsics_resized(orig_w: int, orig_h: int, out_w: int = 518, out_h: int = 392) -> np.ndarray:
    fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
    sx, sy = out_w / float(orig_w), out_h / float(orig_h)
    K = np.array(
        [
            [fx * sx, 0.0, cx * sx],
            [0.0, fy * sy, cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


@torch.no_grad()
def extract_dino_tokens(model: VGGT, image_rgb: np.ndarray) -> np.ndarray:
    """Extract DINO patch tokens as [patch_h, patch_w, 1024]."""
    images_array = np.stack([image_rgb], axis=0)
    vgg_input, patch_w, patch_h = get_vgg_input_imgs(images_array)
    vgg_input = vgg_input.cuda()

    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        if isinstance(out, dict):
            captured["tokens"] = out["x_norm_patchtokens"].detach()
        else:
            captured["tokens"] = out.detach()

    handle = model.aggregator.patch_embed.register_forward_hook(_hook)
    model.update_patch_dimensions(patch_w, patch_h)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _ = model(vgg_input)
    handle.remove()

    tokens = captured["tokens"]
    _, patch_count, dim = tokens.shape
    if patch_h * patch_w != patch_count:
        raise ValueError(f"Patch shape mismatch: {patch_h}*{patch_w} != {patch_count}")
    return tokens[0].float().cpu().numpy().reshape(patch_h, patch_w, dim)


def global_descriptor_from_tokens(dino_tokens: np.ndarray) -> np.ndarray:
    """Compute a compact global semantic descriptor from patch tokens."""
    desc = dino_tokens.reshape(-1, dino_tokens.shape[-1]).mean(axis=0)
    norm = np.linalg.norm(desc) + 1e-8
    return (desc / norm).astype(np.float32)


def build_semantic_cloud_from_frame(frame: FrameData, max_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """Build camera-coordinate cloud and semantic features for one frame."""
    depth = frame.depth_resized_m
    h, w = depth.shape
    yy, xx = np.where(depth > 1e-4)
    if yy.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 1024), dtype=np.float32)

    if yy.size > max_points:
        rng = np.random.RandomState(42)
        sel = rng.choice(yy.size, max_points, replace=False)
        yy, xx = yy[sel], xx[sel]

    z = depth[yy, xx].astype(np.float64)
    K = frame.intrinsics_resized
    x = (xx.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (yy.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)

    patch_y = np.minimum(yy // 14, frame.dino_tokens.shape[0] - 1)
    patch_x = np.minimum(xx // 14, frame.dino_tokens.shape[1] - 1)
    feats = frame.dino_tokens[patch_y, patch_x].astype(np.float32)
    return pts, feats


def merge_submap_frames(
    frames: List[FrameData],
    center_frame: FrameData,
    max_points_per_frame: int = 3500,
    max_total_points: int = 12000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge neighboring frames into the center-frame coordinate system."""
    merged_points: List[np.ndarray] = []
    merged_features: List[np.ndarray] = []
    center_inv = np.linalg.inv(center_frame.pose_c2w)

    for frame in frames:
        pts, feats = build_semantic_cloud_from_frame(frame, max_points=max_points_per_frame)
        if pts.shape[0] == 0:
            continue

        if frame.frame_id != center_frame.frame_id:
            rel = center_inv @ frame.pose_c2w
            pts = (rel[:3, :3] @ pts.T).T + rel[:3, 3][None, :]

        merged_points.append(pts)
        merged_features.append(feats)

    if not merged_points:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 1024), dtype=np.float32)

    points = np.concatenate(merged_points, axis=0)
    features = np.concatenate(merged_features, axis=0)
    if points.shape[0] > max_total_points:
        rng = np.random.RandomState(42)
        sel = rng.choice(points.shape[0], max_total_points, replace=False)
        points = points[sel]
        features = features[sel]
    return points, features


def descriptor_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def to_open3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def prepare_fpfh(points: np.ndarray, voxel: float = 0.05) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd = to_open3d(points)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return pcd, feat


def rotation_error_deg(r_est: np.ndarray, r_gt: np.ndarray) -> float:
    r_diff = r_est @ r_gt.T
    val = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def transform_error(t_est: np.ndarray, t_gt: np.ndarray) -> Tuple[float, float]:
    r_err = rotation_error_deg(t_est[:3, :3], t_gt[:3, :3])
    trans_err = float(np.linalg.norm(t_est[:3, 3] - t_gt[:3, 3]))
    return r_err, trans_err


def chamfer_distance(source: np.ndarray, target: np.ndarray, voxel: float = 0.03) -> float:
    src = to_open3d(source).voxel_down_sample(voxel)
    tgt = to_open3d(target).voxel_down_sample(voxel)
    d1 = np.asarray(src.compute_point_cloud_distance(tgt))
    d2 = np.asarray(tgt.compute_point_cloud_distance(src))
    if d1.size == 0 or d2.size == 0:
        return float("inf")
    return float(np.mean(d1) + np.mean(d2))


def icp_iterative(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    init: np.ndarray,
    point_to_plane: bool,
    max_corr_dist: float,
    max_iters: int,
    trans_eps: float = 1e-4,
    rot_eps_deg: float = 0.05,
) -> Tuple[np.ndarray, int, float]:
    src = to_open3d(src_points).voxel_down_sample(0.03)
    dst = to_open3d(dst_points).voxel_down_sample(0.03)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if point_to_plane
        else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    current = init.copy().astype(np.float64)
    last_rmse = float("inf")
    for step in range(1, max_iters + 1):
        result = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            max_correspondence_distance=max_corr_dist,
            init=current,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )
        new_t = result.transformation
        delta = np.linalg.inv(current) @ new_t
        delta_t = np.linalg.norm(delta[:3, 3])
        delta_r = rotation_error_deg(delta[:3, :3], np.eye(3))
        rmse = float(result.inlier_rmse)
        current = new_t

        if delta_t < trans_eps and delta_r < rot_eps_deg and abs(last_rmse - rmse) < 1e-6:
            return current, step, rmse
        last_rmse = rmse

    return current, max_iters, last_rmse


def multi_scale_icp(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    init: np.ndarray,
    point_to_plane: bool,
    levels: Optional[List[Tuple[float, float, int]]] = None,
) -> Tuple[np.ndarray, int, float]:
    if levels is None:
        levels = [
            (0.08, 0.18, 20),
            (0.05, 0.10, 20),
            (0.03, 0.06, 15),
        ]

    current = init.copy().astype(np.float64)
    total_iters = 0
    last_rmse = float("inf")
    for voxel, corr, max_iters in levels:
        src = to_open3d(src_points).voxel_down_sample(voxel)
        dst = to_open3d(dst_points).voxel_down_sample(voxel)
        if len(src.points) < 50 or len(dst.points) < 50:
            continue
        src_np = np.asarray(src.points)
        dst_np = np.asarray(dst.points)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30))
        dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30))
        current, iters, last_rmse = icp_iterative(
            src_np,
            dst_np,
            init=current,
            point_to_plane=point_to_plane,
            max_corr_dist=corr,
            max_iters=max_iters,
            trans_eps=voxel * 0.15,
            rot_eps_deg=0.08,
        )
        total_iters += iters

    return current, total_iters, last_rmse


def mutual_nn_cosine(feat_a: np.ndarray, feat_b: np.ndarray, subsample: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    idx_a = np.arange(feat_a.shape[0]) if feat_a.shape[0] <= subsample else rng.choice(feat_a.shape[0], subsample, replace=False)
    idx_b = np.arange(feat_b.shape[0]) if feat_b.shape[0] <= subsample else rng.choice(feat_b.shape[0], subsample, replace=False)

    a = torch.from_numpy(feat_a[idx_a]).cuda().float()
    b = torch.from_numpy(feat_b[idx_b]).cuda().float()
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    sim = a @ b.T
    a2b = torch.argmax(sim, dim=1)
    b2a = torch.argmax(sim, dim=0)
    ar = torch.arange(a2b.shape[0], device=a2b.device)
    mutual = b2a[a2b] == ar
    ma = idx_a[ar[mutual].cpu().numpy()]
    mb = idx_b[a2b[mutual].cpu().numpy()]
    return ma, mb


def filtered_semantic_matches(
    src_feat: np.ndarray,
    dst_feat: np.ndarray,
    subsample: int = 6000,
    min_similarity: float = 0.45,
    topk: int = 1500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ma, mb = mutual_nn_cosine(src_feat, dst_feat, subsample=subsample)
    if ma.size == 0:
        return ma, mb, np.zeros((0,), dtype=np.float32)

    src_sel = torch.from_numpy(src_feat[ma]).cuda().float()
    dst_sel = torch.from_numpy(dst_feat[mb]).cuda().float()
    src_sel = torch.nn.functional.normalize(src_sel, dim=1)
    dst_sel = torch.nn.functional.normalize(dst_sel, dim=1)
    scores = torch.sum(src_sel * dst_sel, dim=1).detach().cpu().numpy().astype(np.float32)

    keep = scores >= min_similarity
    if int(keep.sum()) < 8:
        order = np.argsort(scores)[::-1][: min(topk, scores.shape[0])]
        return ma[order], mb[order], scores[order]

    ma = ma[keep]
    mb = mb[keep]
    scores = scores[keep]
    order = np.argsort(scores)[::-1]
    if order.shape[0] > topk:
        order = order[:topk]
    return ma[order], mb[order], scores[order]


def ransac_umeyama(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    n_iter: int = 2000,
    threshold: float = 0.12,
) -> np.ndarray:
    n = src_pts.shape[0]
    if n < 4:
        return np.eye(4, dtype=np.float64)
    rng = np.random.RandomState(42)
    best_inliers = -1
    best_t = np.eye(4, dtype=np.float64)

    for _ in range(n_iter):
        sel = rng.choice(n, 4, replace=False)
        s, r, t = umeyama_alignment(src_pts[sel].T, dst_pts[sel].T, estimate_scale=False)
        pred = (r @ src_pts.T).T + t[None, :]
        inliers = np.linalg.norm(pred - dst_pts, axis=1) < threshold
        count = int(inliers.sum())
        if count > best_inliers:
            best_inliers = count
            if count >= 4:
                s2, r2, t2 = umeyama_alignment(src_pts[inliers].T, dst_pts[inliers].T, estimate_scale=False)
            else:
                s2, r2, t2 = s, r, t
            t_mat = np.eye(4, dtype=np.float64)
            t_mat[:3, :3] = r2
            t_mat[:3, 3] = t2
            best_t = t_mat
    if best_inliers < 4:
        return np.eye(4, dtype=np.float64)
    return best_t


def method_a_fpfh_icp(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, int]:
    src_down, src_feat = prepare_fpfh(src_pts)
    dst_down, dst_feat = prepare_fpfh(dst_pts)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_feat,
        dst_feat,
        mutual_filter=True,
        max_correspondence_distance=0.12,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.12),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(60000, 0.999),
    )
    t_est, iters, _ = multi_scale_icp(
        src_pts,
        dst_pts,
        init=result.transformation,
        point_to_plane=False,
    )
    return t_est, iters


def method_b_dino_coarse(src_pts: np.ndarray, src_feat: np.ndarray, dst_pts: np.ndarray, dst_feat: np.ndarray) -> np.ndarray:
    ma, mb, scores = filtered_semantic_matches(src_feat, dst_feat, subsample=6000, min_similarity=0.50, topk=1200)
    if ma.size < 8:
        return np.eye(4, dtype=np.float64)
    threshold = 0.12 if float(np.mean(scores[: min(64, scores.shape[0])])) > 0.60 else 0.15
    return ransac_umeyama(src_pts[ma], dst_pts[mb], n_iter=2500, threshold=threshold)


def method_c_dino_coarse_icp(src_pts: np.ndarray, src_feat: np.ndarray, dst_pts: np.ndarray, dst_feat: np.ndarray) -> Tuple[np.ndarray, int]:
    coarse = method_b_dino_coarse(src_pts, src_feat, dst_pts, dst_feat)
    fine, iters, _ = multi_scale_icp(
        src_pts,
        dst_pts,
        init=coarse,
        point_to_plane=True,
    )
    return fine, iters


def list_sequence_frames(seq_dir: Path, frame_stride: int) -> List[int]:
    ids = []
    for path in seq_dir.glob("frame-*.color.png"):
        frame = int(path.stem.split(".")[0].split("-")[-1])
        ids.append(frame)
    ids = sorted(ids)
    return ids[::frame_stride]


def make_gt_relative(pose_src: np.ndarray, pose_dst: np.ndarray) -> np.ndarray:
    """Compute source->destination relative transform.

    7-Scenes pose files are treated as world-to-camera (w2c), so:
    T_dst_src = T_dst_w @ T_w_src = T_dst_w @ inv(T_src_w2c)
    """
    return pose_dst @ np.linalg.inv(pose_src)


def compute_baseline(gt_t_dst_src: np.ndarray) -> Tuple[float, float]:
    rot = rotation_error_deg(gt_t_dst_src[:3, :3], np.eye(3))
    trans = float(np.linalg.norm(gt_t_dst_src[:3, 3]))
    return rot, trans


def load_frame(model: VGGT, scene: str, seq: str, seq_dir: Path, frame_id: int) -> FrameData:
    color_path = seq_dir / f"frame-{frame_id:06d}.color.png"
    depth_path = seq_dir / f"frame-{frame_id:06d}.depth.png"
    pose_path = seq_dir / f"frame-{frame_id:06d}.pose.txt"
    image_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to load {color_path}")
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"Failed to load {depth_path}")

    image_resized = resize_rgb_to_model(image_bgr)
    depth_resized = resize_depth_to_model(depth_raw)
    k_resized = intrinsics_resized(orig_w=image_bgr.shape[1], orig_h=image_bgr.shape[0])
    pose = load_pose_c2w(pose_path)
    tokens = extract_dino_tokens(model, image_resized)
    global_desc = global_descriptor_from_tokens(tokens)

    return FrameData(
        scene=scene,
        seq=seq,
        frame_id=frame_id,
        color_path=color_path,
        depth_path=depth_path,
        pose_path=pose_path,
        pose_c2w=pose,
        image_resized=image_resized,
        depth_resized_m=depth_resized,
        intrinsics_resized=k_resized,
        dino_tokens=tokens,
        global_desc=global_desc,
    )


def geometry_points_from_frame(frame: FrameData, max_points: int = 4000) -> np.ndarray:
    depth = frame.depth_resized_m
    h, w = depth.shape
    yy, xx = np.where(depth > 1e-4)
    if yy.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if yy.size > max_points:
        rng = np.random.RandomState(0)
        sel = rng.choice(yy.size, max_points, replace=False)
        yy, xx = yy[sel], xx[sel]
    z = depth[yy, xx].astype(np.float64)
    k = frame.intrinsics_resized
    x = (xx.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (yy.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1)


def build_local_submap(frame_ids: List[int], center_index: int, radius: int) -> List[int]:
    lo = max(0, center_index - radius)
    hi = min(len(frame_ids), center_index + radius + 1)
    return frame_ids[lo:hi]


def collect_pairs(
    model: VGGT,
    dataset_root: Path,
    scene: str,
    frame_stride: int,
    min_rot_deg: float,
    min_trans_m: float,
    max_rot_deg: float,
    max_trans_m: float,
    max_gt_cd: float,
    max_pairs_per_seq: int,
    submap_radius: int,
    min_desc_sim: float,
) -> List[PairData]:
    scene_dir = dataset_root / scene
    seqs = read_test_sequences(scene_dir)
    all_pairs: List[PairData] = []

    for seq in seqs:
        seq_dir = scene_dir / seq
        frame_ids = list_sequence_frames(seq_dir, frame_stride=frame_stride)
        if len(frame_ids) < 2:
            continue

        cache: Dict[int, FrameData] = {}
        candidates: List[Tuple[float, int, int, np.ndarray, float, float]] = []
        frame_to_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        for i in range(len(frame_ids)):
            for j in range(i + 1, len(frame_ids)):
                fi, fj = frame_ids[i], frame_ids[j]
                pose_i = load_pose_c2w(seq_dir / f"frame-{fi:06d}.pose.txt")
                pose_j = load_pose_c2w(seq_dir / f"frame-{fj:06d}.pose.txt")
                gt = make_gt_relative(pose_i, pose_j)
                rot, trans = compute_baseline(gt)
                # Keep large-baseline pairs but enforce upper bounds for overlap feasibility.
                keep_large = (rot >= min_rot_deg or trans >= min_trans_m)
                keep_overlap = (rot <= max_rot_deg and trans <= max_trans_m)
                if keep_large and keep_overlap:
                    score = rot + trans * 50.0
                    candidates.append((score, fi, fj, gt, rot, trans))

        # Deterministic random sampling from all filtered large-baseline pairs,
        # avoiding a bias toward only the most extreme (often non-overlapping) pairs.
        rng = np.random.RandomState(42)
        if len(candidates) > max_pairs_per_seq:
            sel_idx = rng.choice(len(candidates), size=max_pairs_per_seq, replace=False)
            selected = [candidates[int(k)] for k in sel_idx]
        else:
            selected = candidates

        for _, fi, fj, gt, rot, trans in selected:
            if fi not in cache:
                cache[fi] = load_frame(model, scene, seq, seq_dir, fi)
            if fj not in cache:
                cache[fj] = load_frame(model, scene, seq, seq_dir, fj)

            desc_sim = descriptor_similarity(cache[fi].global_desc, cache[fj].global_desc)
            if desc_sim < min_desc_sim:
                continue

            src_sub_ids = build_local_submap(frame_ids, frame_to_index[fi], submap_radius)
            dst_sub_ids = build_local_submap(frame_ids, frame_to_index[fj], submap_radius)
            for sid in src_sub_ids + dst_sub_ids:
                if sid not in cache:
                    cache[sid] = load_frame(model, scene, seq, seq_dir, sid)

            src_submap = [cache[sid] for sid in src_sub_ids]
            dst_submap = [cache[sid] for sid in dst_sub_ids]

            src_geo = geometry_points_from_frame(cache[fi], max_points=3000)
            dst_geo = geometry_points_from_frame(cache[fj], max_points=3000)
            if src_geo.shape[0] < 200 or dst_geo.shape[0] < 200:
                continue
            src_geo_in_dst = (gt[:3, :3] @ src_geo.T).T + gt[:3, 3][None, :]
            gt_cd = chamfer_distance(src_geo_in_dst, dst_geo, voxel=0.04)
            if gt_cd > max_gt_cd:
                continue

            all_pairs.append(
                PairData(
                    scene=scene,
                    seq=seq,
                    src=cache[fi],
                    dst=cache[fj],
                    src_submap=src_submap,
                    dst_submap=dst_submap,
                    gt_t_dst_src=gt,
                    baseline_rot_deg=rot,
                    baseline_trans_m=trans,
                )
            )
    return all_pairs


def evaluate_pair(pair: PairData, methods: Optional[List[str]] = None) -> List[Dict[str, object]]:
    src_pts, src_feat = merge_submap_frames(pair.src_submap, pair.src, max_points_per_frame=2500, max_total_points=10000)
    dst_pts, dst_feat = merge_submap_frames(pair.dst_submap, pair.dst, max_points_per_frame=2500, max_total_points=10000)
    if src_pts.shape[0] < 200 or dst_pts.shape[0] < 200:
        return []

    desc_sim = descriptor_similarity(pair.src.global_desc, pair.dst.global_desc)

    outputs = []
    methods = methods or ["A", "B", "C"]
    for method in methods:
        t0 = time.perf_counter()
        if method == "A":
            t_est, icp_iters = method_a_fpfh_icp(src_pts, dst_pts)
        elif method == "B":
            t_est = method_b_dino_coarse(src_pts, src_feat, dst_pts, dst_feat)
            icp_iters = -1
        else:
            t_est, icp_iters = method_c_dino_coarse_icp(src_pts, src_feat, dst_pts, dst_feat)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        src_aligned = (t_est[:3, :3] @ src_pts.T).T + t_est[:3, 3][None, :]
        cd = chamfer_distance(src_aligned, dst_pts)
        rot_err, trans_err = transform_error(t_est, pair.gt_t_dst_src)
        recall_hit = int((rot_err < 5.0) and (trans_err < 0.05))
        outputs.append(
            {
                "scene": pair.scene,
                "sequence": pair.seq,
                "src_frame": pair.src.frame_id,
                "dst_frame": pair.dst.frame_id,
                "src_submap_size": len(pair.src_submap),
                "dst_submap_size": len(pair.dst_submap),
                "descriptor_similarity": desc_sim,
                "method": method,
                "baseline_rot_deg": pair.baseline_rot_deg,
                "baseline_trans_m": pair.baseline_trans_m,
                "rotation_error_deg": rot_err,
                "translation_error_m": trans_err,
                "recall_hit": recall_hit,
                "icp_iterations": icp_iters,
                "chamfer_distance": cd,
                "runtime_ms": elapsed_ms,
            }
        )
    return outputs


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (scene, method), vals in grouped.items():
        recall = float(np.mean([v["recall_hit"] for v in vals])) if vals else 0.0
        cd = float(np.mean([v["chamfer_distance"] for v in vals])) if vals else float("inf")
        rot = float(np.mean([v["rotation_error_deg"] for v in vals])) if vals else float("inf")
        trans = float(np.mean([v["translation_error_m"] for v in vals])) if vals else float("inf")
        rt = float(np.mean([v["runtime_ms"] for v in vals])) if vals else float("inf")
        iters_vals = [v["icp_iterations"] for v in vals if int(v["icp_iterations"]) > 0]
        icp_iter = float(np.mean(iters_vals)) if iters_vals else 0.0
        summary.append(
            {
                "scene": scene,
                "method": method,
                "num_pairs": len(vals),
                "recall": recall,
                "mean_rotation_error_deg": rot,
                "mean_translation_error_m": trans,
                "mean_chamfer_distance": cd,
                "mean_icp_iterations": icp_iter,
                "mean_runtime_ms": rt,
            }
        )

    for method in ["A", "B", "C"]:
        vals = [r for r in rows if r["method"] == method]
        if not vals:
            continue
        recall = float(np.mean([v["recall_hit"] for v in vals]))
        cd = float(np.mean([v["chamfer_distance"] for v in vals]))
        rot = float(np.mean([v["rotation_error_deg"] for v in vals]))
        trans = float(np.mean([v["translation_error_m"] for v in vals]))
        rt = float(np.mean([v["runtime_ms"] for v in vals]))
        iters_vals = [v["icp_iterations"] for v in vals if int(v["icp_iterations"]) > 0]
        icp_iter = float(np.mean(iters_vals)) if iters_vals else 0.0
        summary.append(
            {
                "scene": "ALL",
                "method": method,
                "num_pairs": len(vals),
                "recall": recall,
                "mean_rotation_error_deg": rot,
                "mean_translation_error_m": trans,
                "mean_chamfer_distance": cd,
                "mean_icp_iterations": icp_iter,
                "mean_runtime_ms": rt,
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    dataset_root = Path("/home/hba/Documents/Dataset/7_scenes")
    output_dir = Path(ROOT_DIR) / "tests" / "tests_result" / "hybrid_registration_7scenes"
    ckpt_path = Path(ROOT_DIR) / "ckpt" / "model_tracker_fixed_e20.pt"

    frame_stride = int(os.environ.get("REG_FRAME_STRIDE", "20"))
    min_rot_deg = float(os.environ.get("REG_MIN_ROT_DEG", "20.0"))
    min_trans_m = float(os.environ.get("REG_MIN_TRANS_M", "0.4"))
    max_rot_deg = float(os.environ.get("REG_MAX_ROT_DEG", "80.0"))
    max_trans_m = float(os.environ.get("REG_MAX_TRANS_M", "2.0"))
    max_gt_cd = float(os.environ.get("REG_MAX_GT_CD", "1.0"))
    max_pairs_per_seq = int(os.environ.get("REG_MAX_PAIRS_PER_SEQ", "25"))
    submap_radius = int(os.environ.get("REG_SUBMAP_RADIUS", "1"))
    min_desc_sim = float(os.environ.get("REG_MIN_DESC_SIM", "0.35"))

    print("=" * 78)
    print("Section 3.3.3 Hybrid Registration Robustness on 7-Scenes")
    print("=" * 78)
    print(f"Dataset: {dataset_root}")
    print(f"Scenes: office, redkitchen")
    print(
        f"Filter: (rot>={min_rot_deg}° OR trans>={min_trans_m}m) "
        f"AND (rot<={max_rot_deg}° AND trans<={max_trans_m}m), frame_stride={frame_stride}"
    )
    print(f"GT-overlap gate: gt_chamfer <= {max_gt_cd}")
    print(f"Descriptor gate: cosine >= {min_desc_sim}, submap_radius={submap_radius}")
    print(f"Max pairs/seq: {max_pairs_per_seq}")

    print("\n[Load model]")
    model = VGGT(merging=None, merge_ratio=0.9, vis_attn_map=False)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    all_rows: List[Dict[str, object]] = []
    for scene in ["office", "redkitchen"]:
        print(f"\n[Collect pairs] scene={scene}")
        pairs = collect_pairs(
            model=model,
            dataset_root=dataset_root,
            scene=scene,
            frame_stride=frame_stride,
            min_rot_deg=min_rot_deg,
            min_trans_m=min_trans_m,
            max_rot_deg=max_rot_deg,
            max_trans_m=max_trans_m,
            max_gt_cd=max_gt_cd,
            max_pairs_per_seq=max_pairs_per_seq,
            submap_radius=submap_radius,
            min_desc_sim=min_desc_sim,
        )
        print(f"  Selected pairs: {len(pairs)}")

        for idx, pair in enumerate(pairs, start=1):
            rows = evaluate_pair(pair)
            all_rows.extend(rows)
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(pairs)} pairs")

    pair_csv = output_dir / "hybrid_registration_pair_results.csv"
    summary_csv = output_dir / "hybrid_registration_summary.csv"

    pair_headers = [
        "scene",
        "sequence",
        "src_frame",
        "dst_frame",
        "src_submap_size",
        "dst_submap_size",
        "descriptor_similarity",
        "method",
        "baseline_rot_deg",
        "baseline_trans_m",
        "rotation_error_deg",
        "translation_error_m",
        "recall_hit",
        "icp_iterations",
        "chamfer_distance",
        "runtime_ms",
    ]
    write_csv(pair_csv, all_rows, pair_headers)

    summary_rows = summarize_rows(all_rows)
    summary_headers = [
        "scene",
        "method",
        "num_pairs",
        "recall",
        "mean_rotation_error_deg",
        "mean_translation_error_m",
        "mean_chamfer_distance",
        "mean_icp_iterations",
        "mean_runtime_ms",
    ]
    write_csv(summary_csv, summary_rows, summary_headers)

    print("\n[Done]")
    print(f"  Pair results: {pair_csv}")
    print(f"  Summary:      {summary_csv}")


if __name__ == "__main__":
    main()
