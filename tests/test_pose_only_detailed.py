"""
Detailed pose-only vs full model comparison with proper isolation.

This script isolates camera head computation from the full model to 
accurately measure the speedup when disabling depth prediction.
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import get_sorted_image_paths, load_images_rgb, get_vgg_input_imgs


def load_model_and_data():
    """Load model and data."""
    # Load model
    CKPT_PATH = Path(ROOT_DIR) / "ckpt" / "model_tracker_fixed_e20.pt"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print(f"\n✓ Loaded checkpoint from {CKPT_PATH}")
    
    # Use default model configuration (matching checkpoint)
    model = VGGT(merging=None, merge_ratio=0.9, vis_attn_map=False)
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda()
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    print(f"✓ Model loaded successfully")
    
    # Load data
    DATA_DIR = Path("/home/hba/Documents/Dataset/ScanNet/scans")
    SCENE = "scene0000_00"
    scene_dir = DATA_DIR / SCENE
    
    images_dir = scene_dir / "color"
    image_paths = get_sorted_image_paths(images_dir)[:10]
    images = load_images_rgb([str(p) for p in image_paths])
    
    images_array = np.stack(images)
    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    vgg_input = vgg_input.cuda()
    
    # Ensure batch dimension [B, S, C, H, W]
    if len(vgg_input.shape) == 4:
        vgg_input = vgg_input.unsqueeze(0)
    
    print(f"✓ Data loaded: {images_array.shape} → {vgg_input.shape}")
    
    return model, vgg_input


def forward_aggregator_only(model, vgg_input, num_runs=30):
    """Test aggregator output generation (no head computation)."""
    print("\n" + "="*70)
    print("TEST: Aggregator Only (No Heads)")
    print("="*70)
    
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            agg_out, patch_idx = model.aggregator(vgg_input)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                agg_out, patch_idx = model.aggregator(vgg_input)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
            if (i+1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
    
    return np.array(times)


def forward_pose_head_only(model, vgg_input, num_runs=30):
    """Test camera head computation (after aggregator)."""
    print("\n" + "="*70)
    print("TEST: Camera Head Only (Input from Aggregator)")
    print("="*70)
    
    model.eval()
    
    # Get aggregator output once
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            agg_out, patch_idx = model.aggregator(vgg_input)
    torch.cuda.synchronize()
    
    # Warm-up
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model.camera_head(agg_out)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model.camera_head(agg_out)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
            if (i+1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
    
    return np.array(times)


def forward_depth_head_only(model, vgg_input, num_runs=30):
    """Test depth head computation (after aggregator)."""
    print("\n" + "="*70)
    print("TEST: Depth Head Only (Input from Aggregator)")
    print("="*70)
    
    model.eval()
    
    # Get aggregator output once
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            agg_out, patch_idx = model.aggregator(vgg_input)
    torch.cuda.synchronize()
    
    # Warm-up
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model.depth_head(agg_out, images=vgg_input, patch_start_idx=patch_idx)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model.depth_head(agg_out, images=vgg_input, patch_start_idx=patch_idx)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
            if (i+1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
    
    return np.array(times)


def forward_full_model(model, vgg_input, num_runs=30):
    """Test full model (aggregator + pose + depth)."""
    print("\n" + "="*70)
    print("TEST: Full Model (Aggregator + Camera + Depth)")
    print("="*70)
    
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(vgg_input)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(vgg_input)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end))
            if (i+1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
    
    return np.array(times)


def main():
    print("="*70)
    print("VGGT Pose-Only Performance Analysis (Detailed)")
    print("="*70)
    
    model, vgg_input = load_model_and_data()
    
    # Run tests
    times_agg = forward_aggregator_only(model, vgg_input, num_runs=30)
    times_pose = forward_pose_head_only(model, vgg_input, num_runs=30)
    times_depth = forward_depth_head_only(model, vgg_input, num_runs=30)
    times_full = forward_full_model(model, vgg_input, num_runs=30)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    S = vgg_input.shape[1]  # Number of sequences
    
    print(f"\nPer-Frame Latency Breakdown ({S} frames per batch):")
    print(f"  Aggregator:            {np.mean(times_agg)/S:7.2f}ms/frame ± {np.std(times_agg)/S:5.2f}ms")
    print(f"  Camera Head:           {np.mean(times_pose)/S:7.2f}ms/frame ± {np.std(times_pose)/S:5.2f}ms")
    print(f"  Depth Head:            {np.mean(times_depth)/S:7.2f}ms/frame ± {np.std(times_depth)/S:5.2f}ms")
    print(f"  Full Model (measured): {np.mean(times_full)/S:7.2f}ms/frame ± {np.std(times_full)/S:5.2f}ms")
    
    # Decomposition
    expected_agg_cam = np.mean(times_agg) + np.mean(times_pose)
    
    print(f"\n[Model Composition Analysis]")
    print(f"  Aggregator:        {np.mean(times_agg):7.2f}ms  ({np.mean(times_agg)/np.mean(times_full)*100:5.1f}% of full)")
    print(f"  Camera Head:       {np.mean(times_pose):7.2f}ms  ({np.mean(times_pose)/np.mean(times_full)*100:5.1f}% of full)")
    print(f"  Depth Head:        {np.mean(times_depth):7.2f}ms  ({np.mean(times_depth)/np.mean(times_full)*100:5.1f}% of full)")
    print(f"  ─────────────────────────────────")
    print(f"  Sum (Agg+Cam+Depth): {np.mean(times_agg)+np.mean(times_pose)+np.mean(times_depth):7.2f}ms")
    print(f"  Full Model (actual): {np.mean(times_full):7.2f}ms")
    print(f"  Overhead:          {np.mean(times_full)-(np.mean(times_agg)+np.mean(times_pose)+np.mean(times_depth)):7.2f}ms")
    
    # Pose-only speedup
    pose_only_time = np.mean(times_agg) + np.mean(times_pose)
    speedup = np.mean(times_full) / pose_only_time
    speedup_percent = (1 - pose_only_time / np.mean(times_full)) * 100
    
    print(f"\n[Pose-Only Mode Analysis]")
    print(f"  Full Model:          {np.mean(times_full):7.2f}ms")
    print(f"  Pose-Only (Agg+Cam): {pose_only_time:7.2f}ms")
    print(f"  Speedup:             {speedup:7.2f}x ({speedup_percent:5.1f}% faster)")
    
    # FPS calculation
    print(f"\n[Frame Rate Analysis]")
    print(f"  Full Model FPS (per frame):   {1000/(np.mean(times_full)/S):.2f} FPS")
    print(f"  Pose-Only FPS (per frame):    {1000/(pose_only_time/S):7.2f} FPS")
    print(f"  FPS Improvement:              {(1000/(pose_only_time/S)) - (1000/(np.mean(times_full)/S)):.2f} FPS")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
