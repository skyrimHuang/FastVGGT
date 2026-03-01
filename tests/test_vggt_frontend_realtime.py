"""
VGGT Front-end Real-time Performance Testing

This script evaluates VGGT as an AR front-end in terms of:
1. Full model inference latency (with depth + pose)
2. Pose-only mode latency (depth module disabled)
3. Per-module timing breakdown
4. Frame rate and memory consumption
5. Comparison with AR real-time requirements (30/60 FPS)

Usage:
    conda run -n fastvggt python tests/test_vggt_frontend_realtime.py
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image
import json
import psutil
import csv

# Setup path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import get_sorted_image_paths, load_images_rgb, get_vgg_input_imgs


class LatencyProfiler:
    """Timing and profiling utility for VGGT inference."""
    
    def __init__(self, name: str = "Unnamed"):
        self.name = name
        self.timings = []
        self.start_event = None
        self.end_event = None
    
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_event = time.perf_counter()
    
    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_event = time.perf_counter()
        elapsed = (self.end_event - self.start_event) * 1000  # Convert to ms
        self.timings.append(elapsed)
        return elapsed
    
    def mean(self):
        return np.mean(self.timings) if self.timings else 0.0
    
    def std(self):
        return np.std(self.timings) if self.timings else 0.0
    
    def min(self):
        return np.min(self.timings) if self.timings else 0.0
    
    def max(self):
        return np.max(self.timings) if self.timings else 0.0
    
    def fps(self):
        """Frames per second"""
        mean = self.mean()
        return 1000 / mean if mean > 0 else 0.0
    
    def reset(self):
        self.timings = []


class VGGTFrontendTesterWithoutDPT(torch.nn.Module):
    """VGGT model with DPT head disabled for pose-only mode."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, images):
        """Forward pass with DPT disabled (only camera pose output)."""
        # Ensure input has batch dimension [B, S, C, H, W]
        if len(images.shape) == 4:
            images = images.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Process through aggregator (includes patch embedding, transformer, etc.)
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
                
                # Camera head only (skip depth head)
                if self.model.camera_head is not None:
                    pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                    return {
                        'pose_enc': pose_enc_list[-1],  # [B, S, 9] - last iteration
                        'pose_enc_list': pose_enc_list,
                        'depth': None,  # Disabled
                    }
                
                return None


def load_scannet_sequence(scene_dir: Path, num_frames: int = 10) -> tuple:
    """Load ScanNet image sequence."""
    images_dir = scene_dir / "color"
    image_paths = get_sorted_image_paths(images_dir)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Limit to num_frames
    image_paths = image_paths[:num_frames]
    
    # Load images
    images = load_images_rgb([str(p) for p in image_paths])
    
    if not images:
        raise ValueError("Failed to load images")
    
    return images, len(images)


def test_full_model_inference(model: VGGT, images: list, num_runs: int = 50) -> dict:
    """Test full VGGT model (with depth + pose)."""
    print("\n" + "="*70)
    print("TEST 1: Full VGGT Model (Depth + Pose)")
    print("="*70)
    
    # Convert images to tensor
    images_array = np.stack(images)  # [N, H, W, 3]
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)  # [1, N, 3, H', W']
    vgg_input = vgg_input.cuda()
    
    print(f"Input shape: {vgg_input.shape}")
    print(f"Patch dimensions: {patch_width}×{patch_height}")
    
    model.eval()
    model = model.cuda()
    # Single type conversion - only float32 needed
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    # Warm-up runs (2x)
    with torch.no_grad():
        for _ in range(2):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(vgg_input)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measurement
    profiler = LatencyProfiler("Full Model")
    memory_snapshots = []
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            profiler.start()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(vgg_input)
            elapsed = profiler.end()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_snapshots.append(peak_mem)
            
            if (i + 1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {elapsed:.2f}ms")
    
    S = vgg_input.shape[1]  # num sequences
    per_frame_latency = profiler.mean() / S
    
    results = {
        'model': 'Full VGGT',
        'num_runs': num_runs,
        'num_sequences': S,
        'total_latency_ms': profiler.mean(),
        'per_frame_latency_ms': per_frame_latency,
        'std_ms': profiler.std(),
        'min_ms': profiler.min(),
        'max_ms': profiler.max(),
        'fps': profiler.fps(),
        'fps_per_frame': 1000 / per_frame_latency if per_frame_latency > 0 else 0.0,
        'peak_memory_mb': np.mean(memory_snapshots),
        'peak_memory_std_mb': np.std(memory_snapshots),
        'timings': profiler.timings,
    }
    
    print(f"\n[Full Model Results]")
    print(f"  Total inference (batch of {S} sequences): {results['total_latency_ms']:.2f}ms ± {results['std_ms']:.2f}ms")
    print(f"  Per-frame latency: {results['per_frame_latency_ms']:.4f}ms")
    print(f"  FPS (per sequence): {results['fps']:.2f}")
    print(f"  FPS (per frame): {results['fps_per_frame']:.2f}")
    print(f"  Peak GPU memory: {results['peak_memory_mb']:.1f}MB ± {results['peak_memory_std_mb']:.1f}MB")
    
    return results


def test_pose_only_mode(model: VGGT, images: list, num_runs: int = 50) -> dict:
    """Test VGGT with DPT module disabled (pose-only mode)."""
    print("\n" + "="*70)
    print("TEST 2: VGGT Pose-Only Mode (DPT Disabled)")
    print("="*70)
    
    # Convert images to tensor
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    vgg_input = vgg_input.cuda()
    
    # Create pose-only model
    pose_only_model = VGGTFrontendTesterWithoutDPT(model)
    pose_only_model.eval()
    pose_only_model = pose_only_model.cuda()
    # Single type conversion
    for param in pose_only_model.model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    print(f"Input shape: {vgg_input.shape}")
    
    # Warm-up runs (2x)
    with torch.no_grad():
        for _ in range(2):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = pose_only_model(vgg_input)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Measurement
    profiler = LatencyProfiler("Pose-Only")
    memory_snapshots = []
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            profiler.start()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = pose_only_model(vgg_input)
            elapsed = profiler.end()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            memory_snapshots.append(peak_mem)
            
            if (i + 1) % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {elapsed:.2f}ms")
    
    S = vgg_input.shape[1]
    per_frame_latency = profiler.mean() / S
    
    results = {
        'model': 'Pose-Only VGGT',
        'num_runs': num_runs,
        'num_sequences': S,
        'total_latency_ms': profiler.mean(),
        'per_frame_latency_ms': per_frame_latency,
        'std_ms': profiler.std(),
        'min_ms': profiler.min(),
        'max_ms': profiler.max(),
        'fps': profiler.fps(),
        'fps_per_frame': 1000 / per_frame_latency if per_frame_latency > 0 else 0.0,
        'peak_memory_mb': np.mean(memory_snapshots),
        'peak_memory_std_mb': np.std(memory_snapshots),
        'timings': profiler.timings,
    }
    
    print(f"\n[Pose-Only Results]")
    print(f"  Total inference (batch of {S} sequences): {results['total_latency_ms']:.2f}ms ± {results['std_ms']:.2f}ms")
    print(f"  Per-frame latency: {results['per_frame_latency_ms']:.4f}ms")
    print(f"  FPS (per sequence): {results['fps']:.2f}")
    print(f"  FPS (per frame): {results['fps_per_frame']:.2f}")
    print(f"  Peak GPU memory: {results['peak_memory_mb']:.1f}MB ± {results['peak_memory_std_mb']:.1f}MB")
    
    return results


def test_module_breakdown(model: VGGT, images: list, num_runs: int = 20) -> dict:
    """Detailed timing breakdown per module - optimized to reduce sync overhead."""
    print("\n" + "="*70)
    print("TEST 3: Per-Module Timing Breakdown")
    print("="*70)
    
    images_array = np.stack(images)
    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    vgg_input = vgg_input.cuda()
    
    # Ensure input has batch dimension [B, S, C, H, W]
    if len(vgg_input.shape) == 4:
        vgg_input = vgg_input.unsqueeze(0)
    
    model.eval()
    model = model.cuda()
    
    # Single type conversion
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    
    # Create CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    module_timings = {
        'Aggregator': [],
        'Camera Head': [],
        'Depth Head': [],
        'Total': [],
    }
    
    # Warm-up
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            aggregated_tokens_list, patch_start_idx = model.aggregator(vgg_input)
            pose_enc_list = model.camera_head(aggregated_tokens_list)
            depth, depth_conf = model.depth_head(aggregated_tokens_list, images=vgg_input, patch_start_idx=patch_start_idx)
    
    torch.cuda.synchronize()

    with torch.no_grad():
        for run in range(num_runs):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Aggregator
                start_event.record()
                aggregated_tokens_list, patch_start_idx = model.aggregator(vgg_input)
                end_event.record()
                torch.cuda.synchronize()
                module_timings['Aggregator'].append(start_event.elapsed_time(end_event))
                
                # Camera Head
                start_event.record()
                pose_enc_list = model.camera_head(aggregated_tokens_list)
                end_event.record()
                torch.cuda.synchronize()
                module_timings['Camera Head'].append(start_event.elapsed_time(end_event))
                
                # Depth Head
                start_event.record()
                depth, depth_conf = model.depth_head(
                    aggregated_tokens_list, 
                    images=vgg_input, 
                    patch_start_idx=patch_start_idx
                )
                end_event.record()
                torch.cuda.synchronize()
                module_timings['Depth Head'].append(start_event.elapsed_time(end_event))
            
            # Total is sum of all modules
            total = module_timings['Aggregator'][-1] + module_timings['Camera Head'][-1] + module_timings['Depth Head'][-1]
            module_timings['Total'].append(total)
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run+1}/{num_runs}")
    
    # Compute statistics
    results = {}
    for module_name, timings in module_timings.items():
        results[module_name] = {
            'mean_ms': np.mean(timings),
            'std_ms': np.std(timings),
            'min_ms': np.min(timings),
            'max_ms': np.max(timings),
        }
    
    print(f"\n[Per-Module Breakdown]")
    for module_name, stats in results.items():
        print(f"  {module_name:25s}: {stats['mean_ms']:6.2f}ms ± {stats['std_ms']:5.2f}ms "
              f"[{stats['min_ms']:6.2f}ms, {stats['max_ms']:6.2f}ms]")
    
    # Calculate Depth Head cost
    depth_cost = results['Depth Head']['mean_ms']
    total_cost = results['Total']['mean_ms']
    depth_ratio = (depth_cost / total_cost) * 100
    
    print(f"\n[Depth Head Analysis]")
    print(f"  Depth Head cost: {depth_cost:.2f}ms ({depth_ratio:.1f}% of total)")
    if total_cost - depth_cost > 0:
        print(f"  Speedup without Depth Head: {total_cost / (total_cost - depth_cost):.2f}x")
    
    return results


def compare_with_requirements(full_results: dict, pose_only_results: dict) -> None:
    """Compare with AR real-time requirements."""
    print("\n" + "="*70)
    print("REQUIREMENT ANALYSIS")
    print("="*70)
    
    AR_REQUIREMENTS = {
        '30 FPS (Mobile AR)': 33.33,
        '60 FPS (VR/High-end AR)': 16.67,
    }
    
    print(f"\n[Full Model]")
    for name, max_latency_ms in AR_REQUIREMENTS.items():
        latency = full_results['per_frame_latency_ms']
        status = "✅ PASS" if latency <= max_latency_ms else "❌ FAIL"
        ratio = latency / max_latency_ms
        print(f"  {name:25s}: {latency:.4f}ms vs {max_latency_ms:.2f}ms {status} ({ratio:.2f}x)")
    
    print(f"\n[Pose-Only Model]")
    for name, max_latency_ms in AR_REQUIREMENTS.items():
        latency = pose_only_results['per_frame_latency_ms']
        status = "✅ PASS" if latency <= max_latency_ms else "❌ FAIL"
        ratio = latency / max_latency_ms
        print(f"  {name:25s}: {latency:.4f}ms vs {max_latency_ms:.2f}ms {status} ({ratio:.2f}x)")
    
    print(f"\n[Comparison]")
    speedup = full_results['per_frame_latency_ms'] / pose_only_results['per_frame_latency_ms']
    print(f"  Pose-only speedup vs Full: {speedup:.2f}x")
    print(f"  Latency reduction: {(1 - pose_only_results['per_frame_latency_ms'] / full_results['per_frame_latency_ms']) * 100:.1f}%")


def generate_comparison_table(full_results: dict, pose_only_results: dict, module_results: dict) -> str:
    """Generate comprehensive comparison table."""
    
    table = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    VGGT Frontend Real-time Performance Report                  ║
╚════════════════════════════════════════════════════════════════════════════════╝

[TABLE 1: Total Inference Latency Comparison]
┌────────────────────────┬──────────────────┬──────────────────┬─────────────────┐
│ Metric                 │ Full Model       │ Pose-Only Model  │ Speedup         │
├────────────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Batch Latency (ms)     │ {:14.2f}  │ {:14.2f}  │ {:13.2f}x  │
│ Per-Frame (ms)         │ {:14.4f}  │ {:14.4f}  │ {:13.2f}x  │
│ FPS (per sequence)     │ {:14.2f}  │ {:14.2f}  │ N/A         │
│ FPS (per frame)        │ {:14.2f}  │ {:14.2f}  │ N/A         │
│ GPU Memory (MB)        │ {:14.1f}  │ {:14.1f}  │ {:13.2f}x  │
└────────────────────────┴──────────────────┴──────────────────┴─────────────────┘

[TABLE 2: Per-Module Timing Breakdown (Full Model)]
┌────────────────────────────────┬───────────┬──────────┬────────┬────────┐
│ Module                         │ Mean (ms) │ Std (ms) │ Min    │ Max    │
├────────────────────────────────┼───────────┼──────────┼────────┼────────┤
""".format(
        full_results['total_latency_ms'],
        pose_only_results['total_latency_ms'],
        full_results['per_frame_latency_ms'] / pose_only_results['per_frame_latency_ms'],
        full_results['per_frame_latency_ms'],
        pose_only_results['per_frame_latency_ms'],
        full_results['per_frame_latency_ms'] / pose_only_results['per_frame_latency_ms'],
        full_results['fps'],
        pose_only_results['fps'],
        full_results['fps_per_frame'],
        pose_only_results['fps_per_frame'],
        full_results['peak_memory_mb'],
        pose_only_results['peak_memory_mb'],
        full_results['peak_memory_mb'] / pose_only_results['peak_memory_mb'],
    )
    
    for module_name, stats in module_results.items():
        if module_name != 'Total':
            table += f"│ {module_name:30s} │ {stats['mean_ms']:9.2f} │ {stats['std_ms']:8.2f} │ {stats['min_ms']:6.2f} │ {stats['max_ms']:6.2f} │\n"
    
    table += f"├────────────────────────────────┼───────────┼──────────┼────────┼────────┤\n"
    table += f"│ {'Total':30s} │ {module_results['Total']['mean_ms']:9.2f} │ {module_results['Total']['std_ms']:8.2f} │ {module_results['Total']['min_ms']:6.2f} │ {module_results['Total']['max_ms']:6.2f} │\n"
    table += "└────────────────────────────────┴───────────┴──────────┴────────┴────────┘\n"
    
    table += "\n[TABLE 3: AR Real-time Requirements Comparison]\n"
    table += "┌──────────────────────────────┬────────────────────┬─────────────┬────────────┐\n"
    table += "│ Requirement                  │ Max Latency (ms)   │ Full Model  │ Pose-Only  │\n"
    table += "├──────────────────────────────┼────────────────────┼─────────────┼────────────┤\n"
    
    requirements = [
        ('Mobile AR (30 FPS)', 33.33),
        ('VR/High-end (60 FPS)', 16.67),
    ]
    
    for req_name, max_latency in requirements:
        full_ok = "✅ PASS" if full_results['per_frame_latency_ms'] <= max_latency else "❌ FAIL"
        pose_ok = "✅ PASS" if pose_only_results['per_frame_latency_ms'] <= max_latency else "❌ FAIL"
        table += f"│ {req_name:28s} │ {max_latency:18.2f} │ {full_ok:11s} │ {pose_ok:10s} │\n"
    
    table += "└──────────────────────────────┴────────────────────┴─────────────┴────────────┘\n"
    
    return table


def save_results_to_json(full_results: dict, pose_only_results: dict, module_results: dict, 
                         output_dir: Path) -> None:
    """Save detailed results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove timings lists for JSON serialization
    full_results_json = {k: v for k, v in full_results.items() if k != 'timings'}
    pose_only_results_json = {k: v for k, v in pose_only_results.items() if k != 'timings'}
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'full_model': full_results_json,
        'pose_only_model': pose_only_results_json,
        'module_breakdown': module_results,
    }
    
    output_file = output_dir / 'vggt_realtime_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


def save_timings_to_csv(full_results: dict, pose_only_results: dict, output_dir: Path) -> None:
    """Save per-run timings to CSV for plotting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'vggt_timings.csv'
    
    max_runs = max(len(full_results['timings']), len(pose_only_results['timings']))
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Full_Model_ms', 'Pose_Only_ms', 'Speedup'])
        
        for i in range(max_runs):
            full_time = full_results['timings'][i] if i < len(full_results['timings']) else None
            pose_time = pose_only_results['timings'][i] if i < len(pose_only_results['timings']) else None
            speedup = full_time / pose_time if full_time and pose_time else None
            
            writer.writerow([
                i + 1,
                f"{full_time:.4f}" if full_time else "",
                f"{pose_time:.4f}" if pose_time else "",
                f"{speedup:.4f}" if speedup else "",
            ])
    
    print(f"✅ Timing data saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = Path("/home/hba/Documents/Dataset/ScanNet/scans")
    SCENE = "scene0000_00"
    CKPT_PATH = "/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt"
    OUTPUT_DIR = Path("./tests/tests_result/vggt_realtime")
    
    print("\n" + "="*70)
    print("VGGT Front-end Real-time Performance Testing")
    print("="*70)
    
    # Load model
    print(f"\n[Loading Model]")
    print(f"  Checkpoint: {CKPT_PATH}")
    model = VGGT(merging=None, merge_ratio=0.9, vis_attn_map=False)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda()
    # Explicitly convert all parameters to float32
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    print(f"  Model loaded successfully (all params converted to float32)")
    
    # Load data
    print(f"\n[Loading Data]")
    scene_dir = DATA_DIR / SCENE
    images, num_frames = load_scannet_sequence(scene_dir, num_frames=10)
    print(f"  Scene: {SCENE}")
    print(f"  Frames loaded: {num_frames}")
    print(f"  Image shape: {images[0].shape}")
    
    # Run tests
    print(f"\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)
    
    full_results = test_full_model_inference(model, images, num_runs=50)
    pose_only_results = test_pose_only_mode(model, images, num_runs=50)
    module_results = test_module_breakdown(model, images, num_runs=20)
    
    # Analysis
    compare_with_requirements(full_results, pose_only_results)
    
    # Generate report
    comparison_table = generate_comparison_table(full_results, pose_only_results, module_results)
    print(comparison_table)
    
    # Save results
    save_results_to_json(full_results, pose_only_results, module_results, OUTPUT_DIR)
    save_timings_to_csv(full_results, pose_only_results, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ Testing completed successfully")
    print("="*70)
