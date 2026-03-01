"""
Shared utilities for AR pipeline feasibility verification tests (Section 3.3.3).

Provides:
- Model loading and initialization
- ScanNet data loading (images, GT poses, GT point clouds)
- DINOv2 feature extraction via forward hook
- Semantic point cloud construction (Feature Lifting: 2D → 3D)
- Timing utilities

Usage:
    conda activate fastvggt
    python tests/test_semantic_pointcloud.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from contextlib import contextmanager

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    umeyama_alignment,
)

# ==================== Constants ====================
DEFAULT_CKPT = os.path.join(ROOT_DIR, "ckpt", "model_tracker_fixed_e20.pt")
DEFAULT_SCANNET_DIR = "/home/hba/Documents/Dataset/ScanNet/scans"
DEFAULT_SCENE = "scene0000_00"
PATCH_SIZE = 14
DINO_FEATURE_DIM = 1024


# ==================== Timing ====================
class Timer:
    """Precision timer with CUDA synchronization support."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, label: str, cuda_sync: bool = True):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[label] = (time.perf_counter() - t0) * 1000.0

    def report(self, title: str = "Timing Report"):
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print(f"{'=' * 65}")
        for label, ms in self.timings.items():
            print(f"  {label:50s} {ms:9.2f} ms")
        print(f"{'=' * 65}\n")

    def get(self, label: str) -> float:
        return self.timings.get(label, 0.0)


# ==================== Model Loading ====================
def load_model(ckpt_path: str = DEFAULT_CKPT, dtype=torch.float16) -> VGGT:
    """Load VGGT model with pretrained weights."""
    print(f"[Model] Loading from {ckpt_path} ...")
    model = VGGT()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    if incompat.missing_keys:
        print(f"  Missing keys ({len(incompat.missing_keys)}): {incompat.missing_keys[:3]}...")
    if incompat.unexpected_keys:
        print(f"  Unexpected keys ({len(incompat.unexpected_keys)}): {incompat.unexpected_keys[:3]}...")
    model = model.cuda().eval().to(dtype)
    print(f"  Model loaded. dtype={dtype}, device=cuda")
    return model


# ==================== Data Loading ====================
def load_scannet_frames(
    scene: str = DEFAULT_SCENE,
    data_dir: str = DEFAULT_SCANNET_DIR,
    max_frames: int = 10,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int, int, List]:
    """
    Load ScanNet scene images and GT poses.

    Returns:
        vgg_input: [S, 3, H, W] tensor in [0, 1]
        gt_c2ws: [S, 4, 4] camera-to-world GT poses (relative to first frame)
        first_gt_pose: [4, 4] original first frame pose in ScanNet world
        patch_width, patch_height: int
        image_paths: list of Path
    """
    scene_dir = Path(data_dir) / scene
    images_dir = scene_dir / "color"
    pose_path = scene_dir / "pose"

    # Load GT poses
    poses_gt, first_gt_pose, available_ids = load_poses(pose_path)
    assert poses_gt is not None, f"Failed to load poses from {pose_path}"

    # Load and select images
    image_paths = get_sorted_image_paths(images_dir)
    selected_ids, selected_paths, selected_indices = build_frame_selection(
        image_paths, available_ids, max_frames
    )

    # Load images → tensor
    images = load_images_rgb(selected_paths)
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)

    # Get GT c2w for selected frames (relative to first frame, as 4×4)
    c2ws_3x4 = poses_gt[selected_indices]  # [S, 4, 4] — load_poses already returns 4×4
    # Ensure 4×4
    if c2ws_3x4.shape[-2:] == (3, 4):
        ones = np.zeros((len(c2ws_3x4), 1, 4))
        ones[:, 0, 3] = 1.0
        c2ws_3x4 = np.concatenate([c2ws_3x4, ones], axis=1)
    gt_c2ws = c2ws_3x4

    S = vgg_input.shape[0]
    print(f"[Data] Loaded {S} frames from {scene}")
    print(f"  Image tensor: {tuple(vgg_input.shape)}, Patches: {patch_width}×{patch_height}")
    return vgg_input, gt_c2ws, first_gt_pose, patch_width, patch_height, list(selected_paths)


def load_gt_pointcloud(scene: str = DEFAULT_SCENE, data_dir: str = DEFAULT_SCANNET_DIR) -> np.ndarray:
    """Load ScanNet GT point cloud from PLY file. Returns [N, 3] float64."""
    import open3d as o3d
    ply_path = Path(data_dir) / scene / f"{scene}_vh_clean_2.ply"
    assert ply_path.exists(), f"GT PLY not found: {ply_path}"
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"[Data] GT point cloud: {points.shape[0]:,} points from {ply_path.name}")
    return points


# ==================== VGGT Inference with DINOv2 Extraction ====================
@torch.no_grad()
def run_vggt_with_dino_features(
    model: VGGT,
    vgg_input: torch.Tensor,
    patch_width: int,
    patch_height: int,
    dtype=torch.float16,
    timer: Optional[Timer] = None,
) -> Dict:
    """
    Run VGGT forward pass and simultaneously extract raw DINOv2 patch tokens
    via a forward hook on the backbone (pre-aggregation, session-invariant).

    Returns dict with:
        'depth':      [S, H, W, 1]        float32 numpy
        'depth_conf': [S, H, W]           float32 numpy
        'extrinsic':  [S, 3, 4]           float32 numpy (camera-from-world / w2c)
        'intrinsic':  [S, 3, 3]           float32 numpy
        'dino_tokens':[S, patch_h, patch_w, 1024]  float32 numpy
        'timings':    dict
    """
    if timer is None:
        timer = Timer()

    if vgg_input.dim() == 4:
        vgg_input = vgg_input.unsqueeze(0)  # [1, S, 3, H, W]
    B, S, C, H, W = vgg_input.shape

    # --- Hook to capture raw DINOv2 patch tokens (before aggregation) ---
    captured = {}

    def _dino_hook(module, inp, out):
        if isinstance(out, dict):
            captured["tokens"] = out["x_norm_patchtokens"].detach()
        else:
            captured["tokens"] = out.detach()

    hook = model.aggregator.patch_embed.register_forward_hook(_dino_hook)

    # Update patch dimensions for dynamic resolution
    model.update_patch_dimensions(patch_width, patch_height)

    # --- Forward pass ---
    with timer.measure("VGGT forward (total)"):
        with torch.cuda.amp.autocast(dtype=dtype):
            images_cuda = vgg_input.cuda().to(torch.bfloat16)
            predictions = model(images_cuda)
    hook.remove()

    # --- Extract DINOv2 tokens ---
    with timer.measure("DINOv2 token reshape"):
        # tokens: [B*S, num_patches, 1024] in bf16
        raw_tokens = captured["tokens"].float()  # → float32 on GPU
        raw_tokens = raw_tokens[:S]  # B=1, take first S
        raw_tokens = raw_tokens.view(S, patch_height, patch_width, DINO_FEATURE_DIM)
        dino_np = raw_tokens.cpu().numpy()  # [S, ph, pw, 1024] float32

    # --- Extract depth ---
    with timer.measure("Depth extraction"):
        depth_np = predictions["depth"][0].detach().float().cpu().numpy()       # [S, H, W, 1]
        depth_conf_np = predictions["depth_conf"][0].detach().float().cpu().numpy()  # [S, H, W]

    # --- Extract pose ---
    with timer.measure("Pose decoding"):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], (H, W)
        )
        extrinsic_np = extrinsic[0].detach().float().cpu().numpy()  # [S, 3, 4]
        intrinsic_np = intrinsic[0].detach().float().cpu().numpy()  # [S, 3, 3]

    # Cleanup GPU
    del predictions, images_cuda, captured
    torch.cuda.empty_cache()

    return {
        "depth": depth_np,
        "depth_conf": depth_conf_np,
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
        "dino_tokens": dino_np,
    }


# ==================== Semantic Point Cloud Construction ====================
def build_semantic_pointcloud(
    depth: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    dino_tokens: np.ndarray,
    depth_conf: Optional[np.ndarray] = None,
    depth_conf_thresh: float = 1.0,
    max_points_per_frame: int = 50000,
    frame_indices: Optional[List[int]] = None,
    timer: Optional[Timer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build semantic point cloud by binding DINOv2 features to 3D points.

    Feature Lifting: For each pixel (u, v) with valid depth, assign the DINOv2
    feature from the corresponding patch at (v // 14, u // 14).

    Args:
        depth:        [S, H, W, 1]
        extrinsic:    [S, 3, 4] camera-from-world (w2c)
        intrinsic:    [S, 3, 3]
        dino_tokens:  [S, patch_h, patch_w, 1024]
        depth_conf:   [S, H, W] (optional)
        depth_conf_thresh: min confidence
        max_points_per_frame: subsample if exceeding
        frame_indices: which frames to use (default: all)

    Returns:
        points:   [N, 3]     float64  world coordinates
        features: [N, 1024]  float32  DINOv2 features
    """
    if timer is None:
        timer = Timer()

    with timer.measure("Semantic PC construction"):
        S_total = depth.shape[0]
        patch_h, patch_w = dino_tokens.shape[1], dino_tokens.shape[2]

        if frame_indices is None:
            frame_indices = list(range(S_total))

        # Filter depth by confidence
        depth_filtered = depth.copy()
        if depth_conf is not None:
            low_conf = depth_conf < depth_conf_thresh
            for fi in frame_indices:
                depth_filtered[fi][low_conf[fi], :] = np.nan

        # Unproject to world 3D — operates on ALL frames but we'll pick relevant ones
        world_points = unproject_depth_map_to_point_map(
            depth_filtered[frame_indices],
            extrinsic[frame_indices],
            intrinsic[frame_indices],
        )  # [len(frame_indices), H, W, 3]

        all_points = []
        all_features = []
        rng = np.random.RandomState(42)

        for local_idx, global_idx in enumerate(frame_indices):
            pts = world_points[local_idx]  # [H, W, 3]

            # Valid mask: no NaN, no Inf, within reasonable bounds
            valid = (
                ~np.isnan(pts[..., 0])
                & ~np.isinf(pts[..., 0])
                & (np.abs(pts[..., 0]) < 50.0)
                & (np.abs(pts[..., 1]) < 50.0)
                & (np.abs(pts[..., 2]) < 50.0)
            )

            v_inds, u_inds = np.where(valid)
            if len(v_inds) == 0:
                continue

            # Subsample for efficiency
            if len(v_inds) > max_points_per_frame:
                sel = rng.choice(len(v_inds), max_points_per_frame, replace=False)
                v_inds, u_inds = v_inds[sel], u_inds[sel]

            # 3D world coordinates
            frame_pts = pts[v_inds, u_inds].astype(np.float64)  # [N, 3]

            # Feature Lifting: pixel (u, v) → patch (v//14, u//14) → DINOv2 feature
            pv = np.minimum(v_inds // PATCH_SIZE, patch_h - 1)
            pu = np.minimum(u_inds // PATCH_SIZE, patch_w - 1)
            frame_feats = dino_tokens[global_idx][pv, pu].astype(np.float32)  # [N, 1024]

            all_points.append(frame_pts)
            all_features.append(frame_feats)

        if not all_points:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, DINO_FEATURE_DIM), dtype=np.float32)

        points = np.concatenate(all_points, axis=0)
        features = np.concatenate(all_features, axis=0)

    print(f"  Semantic PC: {points.shape[0]:,} points × {features.shape[1]}D features "
          f"({timer.get('Semantic PC construction'):.1f} ms)")
    return points, features


# ==================== Feature Matching Utilities ====================
def cosine_match_gpu(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    subsample: int = 5000,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated cosine similarity matching with mutual nearest neighbor.

    Args:
        feat_a: [Na, D] float32
        feat_b: [Nb, D] float32
        subsample: max points per cloud (for memory efficiency)

    Returns:
        idx_a: [M] indices into A of mutual matches
        idx_b: [M] indices into B of mutual matches
        scores: [M] cosine similarity scores
    """
    rng = np.random.RandomState(42)

    Na, Nb = feat_a.shape[0], feat_b.shape[0]
    if Na > subsample:
        sel_a = rng.choice(Na, subsample, replace=False)
    else:
        sel_a = np.arange(Na)
    if Nb > subsample:
        sel_b = rng.choice(Nb, subsample, replace=False)
    else:
        sel_b = np.arange(Nb)

    # Move to GPU, L2-normalize
    a_gpu = torch.from_numpy(feat_a[sel_a]).to(device=device, dtype=torch.float32)
    b_gpu = torch.from_numpy(feat_b[sel_b]).to(device=device, dtype=torch.float32)
    a_norm = F.normalize(a_gpu, dim=1)  # [n_a, D]
    b_norm = F.normalize(b_gpu, dim=1)  # [n_b, D]

    # Cosine similarity via matrix multiply
    sim = a_norm @ b_norm.T  # [n_a, n_b]

    # Mutual nearest neighbors
    nn_a2b_scores, nn_a2b = sim.max(dim=1)   # best B for each A
    nn_b2a_scores, nn_b2a = sim.max(dim=0)   # best A for each B

    arange_a = torch.arange(len(sel_a), device=device)
    mutual_mask = (nn_b2a[nn_a2b] == arange_a)

    idx_a_local = arange_a[mutual_mask].cpu().numpy()
    idx_b_local = nn_a2b[mutual_mask].cpu().numpy()
    scores = nn_a2b_scores[mutual_mask].cpu().numpy()

    # Map back to original indices
    idx_a = sel_a[idx_a_local]
    idx_b = sel_b[idx_b_local]

    del a_gpu, b_gpu, a_norm, b_norm, sim
    torch.cuda.empty_cache()

    return idx_a, idx_b, scores


def ransac_filter(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    n_iter: int = 2000,
    threshold: float = 0.15,
    min_samples: int = 4,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    RANSAC + Umeyama to find robust alignment and filter outliers.

    Finds: A ≈ s * R @ B + t

    Returns: (scale, R, t, inlier_mask, n_inliers)
    """
    N = pts_a.shape[0]
    if N < min_samples:
        return 1.0, np.eye(3), np.zeros(3), np.ones(N, dtype=bool), N

    rng = np.random.RandomState(42)
    best_inliers = 0
    best_mask = np.zeros(N, dtype=bool)
    best_s, best_R, best_t = 1.0, np.eye(3), np.zeros(3)

    for _ in range(n_iter):
        idx = rng.choice(N, min_samples, replace=False)
        s, R, t = umeyama_alignment(
            pts_b[idx].T.astype(np.float64),
            pts_a[idx].T.astype(np.float64),
            estimate_scale=True,
        )

        # Transform all B
        pts_b_t = (s * (R @ pts_b.T) + t[:, None]).T
        dists = np.linalg.norm(pts_a - pts_b_t, axis=1)
        mask = dists < threshold
        n_inl = mask.sum()

        if n_inl > best_inliers:
            best_inliers = n_inl
            best_mask = mask
            best_s, best_R, best_t = s, R, t

    # Refine on all inliers
    if best_inliers >= min_samples:
        best_s, best_R, best_t = umeyama_alignment(
            pts_b[best_mask].T.astype(np.float64),
            pts_a[best_mask].T.astype(np.float64),
            estimate_scale=True,
        )

    return best_s, best_R, best_t, best_mask, best_inliers


# ==================== Transform Utilities ====================
def make_se3_with_scale(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct 4×4 similarity transform from (scale, rotation, translation).
    Transforms points as: p' = s * R @ p + t
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4×4 transform to [N, 3] points."""
    N = points.shape[0]
    homo = np.hstack([points, np.ones((N, 1), dtype=points.dtype)])  # [N, 4]
    return (T @ homo.T).T[:, :3]


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    R_diff = R_est @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")
