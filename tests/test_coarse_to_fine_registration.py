#!/usr/bin/env python3
"""
Test 3: Coarse-to-Fine Registration (Umeyama + ICP)
====================================================
Validates Section 3.3.3 equations (3-14) and (3-15):
  - Coarse: Feature matching + RANSAC + Umeyama → T_init  (eq. 3-14)
  - Fine:   Point-to-Plane ICP on overlap region → ΔT      (eq. 3-15)
  - Global: T_global = ΔT · T_init

Protocol:
  1. Build semantic PCs A and B from the SAME VGGT run (same coordinate system)
  2. Apply known artificial transform T_true to B → B' (simulate coordinate mismatch)
  3. Recover T_est from (A, B') using feature matching → Umeyama → ICP
  4. Compare T_est vs T_true^{-1} (rotation error, translation error, Chamfer dist)
  5. Compare with "naive" ICP (no coarse init) as ablation baseline

Run:
    conda activate fastvggt
    cd /home/hba/Documents/FastVGGT
    python tests/test_coarse_to_fine_registration.py
"""

import os
import sys
import gc
import numpy as np
import torch
import open3d as o3d

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.ar_pipeline_utils import (
    load_model,
    load_scannet_frames,
    run_vggt_with_dino_features,
    build_semantic_pointcloud,
    cosine_match_gpu,
    ransac_filter,
    make_se3_with_scale,
    apply_transform,
    rotation_error_deg,
    Timer,
    print_section,
)


def chamfer_distance_o3d(pts_a: np.ndarray, pts_b: np.ndarray, max_dist: float = 1.0) -> float:
    """Compute symmetric Chamfer distance using Open3D (with voxel downsampling)."""
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(pts_a.astype(np.float64))
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(pts_b.astype(np.float64))

    # Voxel downsample for speed
    voxel = 0.05
    pcd_a = pcd_a.voxel_down_sample(voxel)
    pcd_b = pcd_b.voxel_down_sample(voxel)

    d_a2b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_b2a = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    d_a2b = np.clip(d_a2b, 0, max_dist)
    d_b2a = np.clip(d_b2a, 0, max_dist)
    return float(np.mean(d_a2b) + np.mean(d_b2a))


def icp_registration(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    init_transform: np.ndarray = None,
    max_corr_dist: float = 0.15,
    max_iter: int = 50,
) -> tuple:
    """
    Point-to-Plane ICP using Open3D.

    Returns: (result_transform_4x4, fitness, rmse, time_ms)
    """
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))

    # Voxel downsample for ICP speed
    voxel = 0.03
    src = src.voxel_down_sample(voxel)
    tgt = tgt.voxel_down_sample(voxel)

    # Estimate normals (required for Point-to-Plane)
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if init_transform is None:
        init_transform = np.eye(4)

    import time
    t0 = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_corr_dist,
        init=init_transform.astype(np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    t1 = time.perf_counter()

    return result.transformation, result.fitness, result.inlier_rmse, (t1 - t0) * 1000.0


def generate_artificial_transform(
    rotation_deg: float = 25.0,
    translation: np.ndarray = None,
    scale: float = 1.15,
    seed: int = 42,
) -> np.ndarray:
    """Generate a known Sim(3) transform for testing. Returns 4x4 matrix."""
    rng = np.random.RandomState(seed)

    # Random rotation axis + angle
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.radians(rotation_deg)

    # Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    if translation is None:
        translation = np.array([0.5, -0.3, 0.8])

    T = make_se3_with_scale(scale, R, translation)
    return T


def test_coarse_to_fine_registration(
    scene: str = "scene0000_00",
    max_frames: int = 10,
):
    timer = Timer()

    # ── 1. Setup ──
    print_section("1. Setup")
    with timer.measure("Model loading"):
        model = load_model()
    with timer.measure("Data loading"):
        vgg_input, gt_c2ws, first_gt_pose, pw, ph, paths = load_scannet_frames(
            scene=scene, max_frames=max_frames
        )
    S = vgg_input.shape[0]
    mid = S // 2

    # ── 2. VGGT Inference ──
    print_section("2. VGGT Inference")
    results = run_vggt_with_dino_features(model, vgg_input, pw, ph, timer=timer)

    # Free model to save GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Build Semantic Point Clouds ──
    print_section("3. Build Semantic Point Clouds")
    pts_a, feat_a = build_semantic_pointcloud(
        results["depth"], results["extrinsic"], results["intrinsic"],
        results["dino_tokens"], results["depth_conf"],
        max_points_per_frame=30000, frame_indices=list(range(mid)), timer=timer,
    )
    pts_b, feat_b = build_semantic_pointcloud(
        results["depth"], results["extrinsic"], results["intrinsic"],
        results["dino_tokens"], results["depth_conf"],
        max_points_per_frame=30000, frame_indices=list(range(mid, S)), timer=timer,
    )
    print(f"  PC A: {pts_a.shape[0]:,} pts | PC B: {pts_b.shape[0]:,} pts")

    # ── 4. Apply Artificial Transform to B ──
    print_section("4. Apply Artificial Transform to B → B'")
    T_true = generate_artificial_transform(rotation_deg=25.0, scale=1.15)
    T_true_inv = np.linalg.inv(T_true)

    R_true = T_true[:3, :3]
    t_true = T_true[:3, 3]
    # Extract scale from R (scale * R_pure)
    s_det = np.linalg.det(R_true)
    s_true = np.cbrt(np.abs(s_det))
    R_pure = R_true / s_true

    print(f"  T_true: rotation={25.0}°, translation=‖{np.linalg.norm(t_true):.3f}‖m, scale={s_true:.4f}")

    pts_b_prime = apply_transform(pts_b, T_true)  # B' = T_true(B)
    # Features stay the same — only positions change
    feat_b_prime = feat_b

    # Verify: B' is now misaligned with A
    cd_before = chamfer_distance_o3d(pts_a, pts_b_prime, max_dist=2.0)
    print(f"  Chamfer distance (A vs B'): {cd_before:.4f} m (misaligned)")

    # ── 5. Feature Matching on (A, B') ──
    print_section("5. Feature Matching: A ↔ B' (cosine similarity)")
    with timer.measure("Feature matching"):
        idx_a, idx_b, scores = cosine_match_gpu(feat_a, feat_b_prime, subsample=5000)
    print(f"  Mutual matches: {len(idx_a)}")
    if len(idx_a) > 0:
        print(f"  Cosine similarity: mean={scores.mean():.4f}, min={scores.min():.4f}")

    matched_a = pts_a[idx_a]
    matched_b_prime = pts_b_prime[idx_b]

    # ── 6. RANSAC + Umeyama Coarse Registration (Eq. 3-14) ──
    print_section("6. Coarse Registration: RANSAC + Umeyama (Eq. 3-14)")
    with timer.measure("RANSAC + Umeyama"):
        s_est, R_est, t_est, inlier_mask, n_inliers = ransac_filter(
            matched_a, matched_b_prime,
            n_iter=3000, threshold=0.2, min_samples=4,
        )
    print(f"  RANSAC inliers: {n_inliers} / {len(idx_a)} ({n_inliers / max(len(idx_a), 1) * 100:.1f}%)")
    print(f"  Estimated: scale={s_est:.4f}, ‖t‖={np.linalg.norm(t_est):.4f} m")

    # T_init: transforms B' → approximately aligned with A
    T_init = make_se3_with_scale(s_est, R_est, t_est)
    pts_b_coarse = apply_transform(pts_b_prime, T_init)  # coarse-aligned B

    cd_coarse = chamfer_distance_o3d(pts_a, pts_b_coarse, max_dist=2.0)
    print(f"  Chamfer after coarse: {cd_coarse:.4f} m")

    # Evaluate coarse registration accuracy
    # Expected: T_init ≈ T_true^{-1}, so T_init @ T_true ≈ I
    T_composition = T_init @ T_true
    rot_err_coarse = rotation_error_deg(T_composition[:3, :3] / np.cbrt(np.abs(np.linalg.det(T_composition[:3, :3]))),
                                         np.eye(3))
    trans_err_coarse = np.linalg.norm(T_composition[:3, 3])
    print(f"  Coarse accuracy: rot_err={rot_err_coarse:.2f}°, trans_err={trans_err_coarse:.4f} m")

    # ── 7. ICP Fine Registration (Eq. 3-15) ──
    print_section("7. Fine Registration: Point-to-Plane ICP (Eq. 3-15)")
    with timer.measure("ICP fine registration"):
        delta_T, fitness, rmse, icp_ms = icp_registration(
            pts_b_coarse, pts_a,
            init_transform=np.eye(4),  # already coarse-aligned
            max_corr_dist=0.15, max_iter=50,
        )
    print(f"  ICP fitness: {fitness:.4f}, RMSE: {rmse:.6f} m, time: {icp_ms:.1f} ms")

    # T_global = ΔT · T_init (Eq. 3-15)
    T_global = delta_T @ T_init
    pts_b_final = apply_transform(pts_b_prime, T_global)

    cd_fine = chamfer_distance_o3d(pts_a, pts_b_final, max_dist=2.0)
    print(f"  Chamfer after fine: {cd_fine:.4f} m")

    # Evaluate full pipeline accuracy
    T_comp_fine = T_global @ T_true
    s_comp = np.cbrt(np.abs(np.linalg.det(T_comp_fine[:3, :3])))
    R_comp_pure = T_comp_fine[:3, :3] / s_comp if s_comp > 1e-6 else T_comp_fine[:3, :3]
    rot_err_fine = rotation_error_deg(R_comp_pure, np.eye(3))
    trans_err_fine = np.linalg.norm(T_comp_fine[:3, 3])
    scale_err_fine = abs(s_comp - 1.0)
    print(f"  Full pipeline accuracy:")
    print(f"    Rotation error:    {rot_err_fine:.4f}°")
    print(f"    Translation error: {trans_err_fine:.6f} m")
    print(f"    Scale error:       {scale_err_fine:.6f}")

    # ── 8. Ablation: Naive ICP (no coarse init) ──
    print_section("8. Ablation: Naive ICP (no coarse initialization)")
    with timer.measure("Naive ICP (identity init)"):
        delta_T_naive, fitness_naive, rmse_naive, icp_ms_naive = icp_registration(
            pts_b_prime, pts_a,
            init_transform=np.eye(4),  # identity — no coarse init!
            max_corr_dist=0.5,  # wider search needed
            max_iter=100,
        )
    pts_b_naive = apply_transform(pts_b_prime, delta_T_naive)
    cd_naive = chamfer_distance_o3d(pts_a, pts_b_naive, max_dist=2.0)
    print(f"  Naive ICP: fitness={fitness_naive:.4f}, RMSE={rmse_naive:.4f}, "
          f"Chamfer={cd_naive:.4f} m, time={icp_ms_naive:.1f} ms")
    print(f"  (Expected: high Chamfer — ICP fails without proper initialization)")

    # ── 9. Comparison Table ──
    print_section("9. Registration Comparison")
    header = f"  {'Method':<35} {'Chamfer (m)':<15} {'Rot err (°)':<15} {'Time (ms)'}"
    dashes = f"  {'─' * 80}"
    line1 = f"  {'Before (misaligned B prime)':<35} {cd_before:<15.4f} {'N/A':<15} {'N/A'}"
    line2 = (f"  {'Umeyama coarse only':<35} {cd_coarse:<15.4f} {rot_err_coarse:<15.2f} "
             f"{timer.get('RANSAC + Umeyama'):.1f}")
    line3 = (f"  {'Umeyama + ICP (our pipeline)':<35} {cd_fine:<15.4f} {rot_err_fine:<15.4f} "
             f"{timer.get('RANSAC + Umeyama') + timer.get('ICP fine registration'):.1f}")
    line4 = f"  {'Naive ICP (no coarse, ablation)':<35} {cd_naive:<15.4f} {'N/A':<15} {icp_ms_naive:.1f}"
    
    print(header)
    print(dashes)
    print(line1)
    print(line2)
    print(line3)
    print(line4)

    # ── 10. Validation ──
    print_section("10. Validation Summary")
    # Note: Rotation and translation errors are relative to the large artificial
    # transform (25° rotation + 1.15× scale +1m translation).  The key validation
    # is that coarse-fine is better than naive ICP, and iteration improves accuracy.
    validations = {
        "Coarse chamfer < initial chamfer":         cd_coarse < cd_before,
        "Fine improves or near coarse chamfer":     cd_fine <= (cd_coarse * 1.05),
        "Pipeline rot error < 15°":                 rot_err_fine < 15.0,
        "Pipeline trans error < 0.5 m":             trans_err_fine < 0.5,
        "Pipeline outperforms naive ICP":           cd_fine < cd_naive,
        "Feature matching found > 50 matches":      len(idx_a) > 50,
    }
    all_pass = True
    for name, ok in validations.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    timer.report("Test 3: Coarse-to-Fine Registration — Timing")

    if all_pass:
        print("  *** ALL CHECKS PASSED — Coarse-to-fine registration pipeline is feasible ***\n")
    else:
        print("  *** SOME CHECKS FAILED — review output above ***\n")

    gc.collect()
    torch.cuda.empty_cache()
    return all_pass


if __name__ == "__main__":
    torch.manual_seed(42)
    success = test_coarse_to_fine_registration(scene="scene0000_00", max_frames=10)
    sys.exit(0 if success else 1)
