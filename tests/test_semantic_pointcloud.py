#!/usr/bin/env python3
"""
Test 1: DINOv2 Semantic Point Cloud Construction Verification
=============================================================
Validates Section 3.3.3 prerequisite: 2D DINOv2 features can be "lifted"
to 3D via depth unprojection, producing a semantic point cloud (XYZ + Feature).

Verifies:
  1. DINOv2 raw patch tokens are extractable via forward hook [S, ph, pw, 1024]
  2. Depth maps are pixel-aligned and produce valid 3D world coordinates
  3. Feature Lifting: each 3D point receives a 1024-d DINOv2 feature
  4. Feature vectors have meaningful L2 norms (not degenerate)
  5. Nearby 3D points share similar features (spatial coherence)
  6. Timing breakdown for each pipeline stage

Run:
    conda activate fastvggt
    cd /home/hba/Documents/FastVGGT
    python tests/test_semantic_pointcloud.py
"""

import os
import sys
import gc
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.ar_pipeline_utils import (
    load_model,
    load_scannet_frames,
    run_vggt_with_dino_features,
    build_semantic_pointcloud,
    Timer,
    print_section,
    PATCH_SIZE,
    DINO_FEATURE_DIM,
)


def test_semantic_pointcloud(
    scene: str = "scene0000_00",
    max_frames: int = 5,
):
    timer = Timer()

    # ── 1. Load Model ──
    print_section("1. Model Loading")
    with timer.measure("Model loading"):
        model = load_model()

    # ── 2. Load Data ──
    print_section("2. Data Loading")
    with timer.measure("Data loading"):
        vgg_input, gt_c2ws, first_gt_pose, pw, ph, paths = load_scannet_frames(
            scene=scene, max_frames=max_frames
        )
    S, C, H, W = vgg_input.shape
    expected_ph, expected_pw = H // PATCH_SIZE, W // PATCH_SIZE
    print(f"  Expected patches: {expected_pw}×{expected_ph} = {expected_pw * expected_ph}")

    # ── 3. VGGT Inference + DINOv2 Extraction ──
    print_section("3. VGGT Inference + DINOv2 Feature Extraction")
    results = run_vggt_with_dino_features(model, vgg_input, pw, ph, timer=timer)

    dino_tokens = results["dino_tokens"]  # [S, ph, pw, 1024]
    depth = results["depth"]              # [S, H, W, 1]
    depth_conf = results["depth_conf"]    # [S, H, W]
    extrinsic = results["extrinsic"]      # [S, 3, 4]
    intrinsic = results["intrinsic"]      # [S, 3, 3]

    # ── 4. Shape Verification ──
    print_section("4. Shape Verification")
    checks = {
        "dino_tokens shape": (dino_tokens.shape == (S, expected_ph, expected_pw, DINO_FEATURE_DIM)),
        "depth shape":       (depth.shape == (S, H, W, 1)),
        "depth_conf shape":  (depth_conf.shape == (S, H, W)),
        "extrinsic shape":   (extrinsic.shape == (S, 3, 4)),
        "intrinsic shape":   (intrinsic.shape == (S, 3, 3)),
        "dino dtype float32": (dino_tokens.dtype == np.float32),
        "depth dtype float32": (depth.dtype == np.float32),
    }
    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    # ── 5. Build Semantic Point Cloud ──
    print_section("5. Semantic Point Cloud Construction (Feature Lifting)")
    points, features = build_semantic_pointcloud(
        depth, extrinsic, intrinsic, dino_tokens,
        depth_conf=depth_conf, depth_conf_thresh=1.0,
        max_points_per_frame=30000, timer=timer,
    )

    N = points.shape[0]
    print(f"  Total 3D points: {N:,}")
    print(f"  Points shape:    {points.shape}  dtype={points.dtype}")
    print(f"  Features shape:  {features.shape}  dtype={features.dtype}")

    # ── 6. Feature Quality Analysis ──
    print_section("6. Feature Quality Analysis")
    feat_norms = np.linalg.norm(features, axis=1)
    print(f"  Feature L2 norm — min: {feat_norms.min():.4f}, max: {feat_norms.max():.4f}, "
          f"mean: {feat_norms.mean():.4f}, std: {feat_norms.std():.4f}")

    nonzero_ratio = (feat_norms > 1e-6).sum() / N
    print(f"  Non-zero features: {nonzero_ratio * 100:.2f}%")

    # Spatial coherence: nearby 3D points should have similar features
    with timer.measure("Spatial coherence check"):
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, min(1000, N), replace=False)
        sample_pts = points[sample_idx]
        sample_feats = features[sample_idx]

        tree = cKDTree(sample_pts)
        # For each sampled point, find its 5 nearest neighbors
        dists, nn_idx = tree.query(sample_pts, k=6)  # k=6 because first is self

        # Compute feature cosine similarity with nearest neighbor (exclude self)
        cos_sims = []
        for i in range(len(sample_idx)):
            f_self = sample_feats[i]
            f_nn = sample_feats[nn_idx[i, 1]]  # nearest non-self neighbor
            norm_product = np.linalg.norm(f_self) * np.linalg.norm(f_nn)
            if norm_product > 1e-8:
                cos_sims.append(np.dot(f_self, f_nn) / norm_product)
        cos_sims = np.array(cos_sims)

    print(f"  Spatial coherence (cosine sim of 3D-nearest neighbors):")
    print(f"    mean: {cos_sims.mean():.4f}, median: {np.median(cos_sims):.4f}, "
          f"std: {cos_sims.std():.4f}")
    print(f"    > 0.8: {(cos_sims > 0.8).mean() * 100:.1f}%, "
          f"> 0.5: {(cos_sims > 0.5).mean() * 100:.1f}%")

    # ── 7. Validation Assertions ──
    print_section("7. Validation Summary")
    validations = {
        "Point cloud non-empty":         N > 0,
        "All features non-zero":         nonzero_ratio > 0.99,
        "Feature dim = 1024":            features.shape[1] == DINO_FEATURE_DIM,
        "Points dtype float64":          points.dtype == np.float64,
        "Features dtype float32":        features.dtype == np.float32,
        "Spatial coherence mean > 0.5":  cos_sims.mean() > 0.5,
        "Point coords finite":          np.all(np.isfinite(points)),
        "Feature norms reasonable":      feat_norms.mean() > 0.1,
    }
    for name, ok in validations.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    # ── 8. Timing Report ──
    timer.report("Test 1: Semantic Point Cloud — Timing")

    # ── 9. Memory footprint ──
    print_section("8. Memory Footprint")
    pts_mb = points.nbytes / 1e6
    feat_mb = features.nbytes / 1e6
    print(f"  3D points:  {pts_mb:.1f} MB ({N:,} × 3 × float64)")
    print(f"  Features:   {feat_mb:.1f} MB ({N:,} × 1024 × float32)")
    print(f"  Total:      {pts_mb + feat_mb:.1f} MB")

    if all_pass:
        print("\n  *** ALL CHECKS PASSED — DINOv2 semantic point cloud is fully feasible ***\n")
    else:
        print("\n  *** SOME CHECKS FAILED — review output above ***\n")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_pass


if __name__ == "__main__":
    torch.manual_seed(42)
    success = test_semantic_pointcloud(scene="scene0000_00", max_frames=5)
    sys.exit(0 if success else 1)
