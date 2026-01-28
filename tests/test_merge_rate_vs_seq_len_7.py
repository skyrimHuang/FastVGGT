
import os
import sys
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import SevenScenes
from utils import accuracy, completion
from vggt.models.vggt import VGGT
from criterion import Regr3D_t_ScaleShiftInv, L21

# --- Experiment Configuration ---
# Note: These are placeholders. Replace with the actual paths on your system.
SEVEN_SCENES_ROOT = "path/to/your/7-scenes-dataset"
CKPT_PATH = "path/to/your/model_checkpoint.pt"
OUTPUT_DIR = "e:/GraduateProject/FastVGGT/tests/results"

# --- Global Variables for Testing ---
# Modify this to test different data types
# torch.bfloat16: Better precision, requires Ampere or newer GPU
# torch.float16: Wider GPU compatibility
# torch.float32: Full precision, slower
GLOBAL_DTYPE = torch.float16

# --- Experiment Parameters ---
SEQUENCE_LENGTHS = [5, 10, 30, 50, 100]
MERGE_RATIOS = [0.3, 0.6, 0.9]

def get_args_parser():
    parser = argparse.ArgumentParser("Test Merge Rate vs. Sequence Length", add_help=False)
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--size", type=int, default=518, help="Image size for the model")
    parser.add_argument("--kf", type=int, default=2, help="Keyframe interval")
    parser.add_argument("--use_proj", action="store_true", help="Use Umeyama alignment instead of only ICP")
    return parser

def main(args):
    """
    Main function to run the experiment.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # --- Data Loading ---
    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        # Add other resolutions if needed, or raise an error.
        raise NotImplementedError(f"Resolution for size {args.size} not implemented.")

    # --- Model Loading ---
    # Initial dummy values, will be updated in the loop
    model = VGGT(merging=0, merge_ratio=0.9, enable_point=True) 
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    except FileNotFoundError:
        print(f"Checkpoint file not found at: {CKPT_PATH}")
        print("Please update the CKPT_PATH variable in the script.")
        return

    # --- Determine Data Type ---    
    # Use global dtype but override based on GPU capability if needed
    dtype = GLOBAL_DTYPE
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print("WARNING: bfloat16 not supported on this GPU, falling back to float16")
        dtype = torch.float16
    
    model = model.to(args.device).eval().to(dtype)
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    # --- Experiment Loop ---
    for seq_len in SEQUENCE_LENGTHS:
        for merge_ratio in MERGE_RATIOS:
            print(f"Running experiment with Sequence Length: {seq_len}, Merge Ratio: {merge_ratio}")

            # Update model's merge ratio
            model.merging = 1 # Enable merging
            model.merge_ratio = merge_ratio

            # --- Dataset ---
            try:
                dataset = SevenScenes(
                    split="test",
                    ROOT=SEVEN_SCENES_ROOT,
                    resolution=resolution,
                    num_seq=seq_len,
                    full_video=True,
                    kf_every=args.kf,
                )
            except Exception as e:
                print(f"Error loading 7-Scenes dataset from: {SEVEN_SCENES_ROOT}")
                print(f"Error: {e}")
                print("Please update the SEVEN_SCENES_ROOT variable in the script.")
                return

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=default_collate)

            total_acc = 0
            total_acc_med = 0
            total_comp = 0
            total_comp_med = 0
            total_nc1 = 0
            total_nc1_med = 0
            total_nc2 = 0
            total_nc2_med = 0
            total_runtime = 0
            num_samples = 0

            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"SeqLen {seq_len}, Merge {merge_ratio}"):
                    # Move data to device and handle nested structures
                    for view in batch:
                        for name, tensor in view.items():
                            if isinstance(tensor, torch.Tensor):
                                view[name] = tensor.to(args.device, non_blocking=True)
                            elif isinstance(tensor, (list, tuple)):
                                view[name] = [x.to(args.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in tensor]

                    # --- Image Normalization with Autocast ---
                    with torch.cuda.amp.autocast(dtype=dtype):
                        # Normalize images
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0
                        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)

                    # --- Inference and Timing with Autocast ---
                    with torch.cuda.amp.autocast(dtype=dtype):
                        torch.cuda.synchronize()
                        start_time = time.time()

                        preds_raw = model(imgs_tensor)

                        torch.cuda.synchronize()
                        end_time = time.time()
                        runtime = (end_time - start_time) * 1000  # in ms

                    # --- Process Model Outputs and Evaluation with Autocast ---
                    with torch.cuda.amp.autocast(dtype=dtype):
                        # Reconstruct predictions to match criterion expectation
                        predictions = preds_raw
                        if "pose_enc" in predictions:
                            B, S = predictions["pose_enc"].shape[:2]
                        elif "world_points" in predictions:
                            B, S = predictions["world_points"].shape[:2]
                        else:
                            # Fallback or error if keys missing
                            print("Warning: predictions missing expected keys")
                            continue

                        ress = []
                        for s in range(S):
                            res = {
                                "pts3d_in_other_view": predictions["world_points"][:, s],
                                "conf": predictions["world_points_conf"][:, s],
                                "depth": predictions["depth"][:, s],
                                "depth_conf": predictions["depth_conf"][:, s],
                                "camera_pose": predictions["pose_enc"][:, s, :],
                            }
                            # Add valid_mask if available in batch
                            if isinstance(batch, list) and s < len(batch) and "valid_mask" in batch[s]:
                                res["valid_mask"] = batch[s]["valid_mask"]
                            
                            ress.append(res)
                        
                        preds = ress

                        # --- Evaluation ---
                        # get_all_pts3d_t extracts and aligns ground truth and predictions
                        gt_pts, pred_pts, _, _, _, _ = criterion.get_all_pts3d_t(batch, preds)

                    # --- Per-View Processing (Cropping & Filtering) ---
                    # Replicating logic from eval_7andN.py
                    pts_all_list = []
                    pts_gt_all_list = []
                    
                    for j, view in enumerate(batch):
                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]
                        
                        pts = pred_pts[j].cpu().numpy()[0]
                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        # Center Cropping 224x224 (hardcoded in eval script)
                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        
                        # Apply crop
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]
                        mask = mask[t:b, l:r]

                        pts_all_list.append(pts[None, ...])
                        pts_gt_all_list.append(pts_gt[None, ...])

                    # Concatenate all views
                    pts_all = np.concatenate(pts_all_list, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all_list, axis=0)
                    
                    # Flatten to point cloud (N, 3)
                    pts_all_masked = pts_all.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all.reshape(-1, 3)

                    # Filter non-finite values
                    pts_all_masked = pts_all_masked[np.isfinite(pts_all_masked).all(axis=1)]
                    pts_gt_all_masked = pts_gt_all_masked[np.isfinite(pts_gt_all_masked).all(axis=1)]

                    # Sampling if too many points (from eval_7andN.py)
                    if pts_all_masked.shape[0] > 999999:
                        sample_indices = np.random.choice(
                            pts_all_masked.shape[0], 999999, replace=False
                        )
                        pts_all_masked = pts_all_masked[sample_indices]

                    if pts_gt_all_masked.shape[0] > 999999:
                        sample_indices_gt = np.random.choice(
                            pts_gt_all_masked.shape[0], 999999, replace=False
                        )
                        pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                    if pts_all_masked.shape[0] == 0 or pts_gt_all_masked.shape[0] == 0:
                        # print("Warning: No valid points found for a sample after masking. Skipping.")
                        continue

                    # --- Umeyama Alignment (from eval_7andN.py) ---
                    if args.use_proj:
                        def umeyama_alignment(
                            src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                        ):
                            assert src.shape == dst.shape
                            N, dim = src.shape

                            mu_src = src.mean(axis=0)
                            mu_dst = dst.mean(axis=0)
                            src_c = src - mu_src
                            dst_c = dst - mu_dst

                            Sigma = dst_c.T @ src_c / N  # (3,3)

                            U, D, Vt = np.linalg.svd(Sigma)

                            S = np.eye(dim)
                            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                                S[-1, -1] = -1

                            R = U @ S @ Vt

                            if with_scale:
                                var_src = (src_c**2).sum() / N
                                s = (D * S.diagonal()).sum() / var_src
                            else:
                                s = 1.0

                            t = mu_dst - s * R @ mu_src

                            return s, R, t

                        s, R, t = umeyama_alignment(
                            pts_all_masked, pts_gt_all_masked, with_scale=True
                        )
                        pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                        pts_all_masked = pts_all_aligned

                    # --- Point Cloud Registration and Metrics ---
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked)

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)

                    # Using ICP for alignment
                    threshold = 0.1
                    trans_init = np.eye(4)
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd, pcd_gt, threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())

                    pcd.transform(reg_p2p.transformation)

                    # Calculate accuracy and completion
                    # Note: eval_7andN.py uses normals for metrics if available, 
                    # but simple accuracy/completion function handles None normals.
                    # We pass None for normals as we didn't estimate them on the fly 
                    # (though eval_7andN.py does pcd.estimate_normals()).
                    # Let's add normal estimation to be strictly consistent.
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()
                    
                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

                    total_acc += acc
                    total_acc_med += acc_med
                    total_comp += comp
                    total_comp_med += comp_med
                    total_nc1 += nc1
                    total_nc1_med += nc1_med
                    total_nc2 += nc2
                    total_nc2_med += nc2_med
                    total_runtime += runtime
                    num_samples += 1

            if num_samples > 0:
                avg_acc = total_acc / num_samples
                avg_acc_med = total_acc_med / num_samples
                avg_comp = total_comp / num_samples
                avg_comp_med = total_comp_med / num_samples
                avg_nc1 = total_nc1 / num_samples
                avg_nc1_med = total_nc1_med / num_samples
                avg_nc2 = total_nc2 / num_samples
                avg_nc2_med = total_nc2_med / num_samples
                avg_runtime = total_runtime / num_samples

                print(f"Avg Acc: {avg_acc:.4f}, Avg Comp: {avg_comp:.4f}, Avg NC1: {avg_nc1:.4f}, Avg NC2: {avg_nc2:.4f}, Avg Runtime: {avg_runtime:.2f}ms")

                results.append({
                    "sequence_length": seq_len,
                    "merge_ratio": merge_ratio,
                    "accuracy": avg_acc,
                    "accuracy_median": avg_acc_med,
                    "completion": avg_comp,
                    "completion_median": avg_comp_med,
                    "normal_consistency1": avg_nc1,
                    "normal_consistency1_median": avg_nc1_med,
                    "normal_consistency2": avg_nc2,
                    "normal_consistency2_median": avg_nc2_med,
                    "runtime_ms": avg_runtime
                })

    # --- Save Results to CSV ---
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "merge_rate_vs_seq_len_results.csv")
        # index=False saves the header (column names) but not the row numbers
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
