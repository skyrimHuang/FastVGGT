"""
Norm-guided Partition Ratio Optimization
=========================================

Test different partition ratios to find optimal balance between:
- Geometry preservation (accuracy)
- Token compression (efficiency)

IMPORTANT: This script forces fresh evaluation every run by clearing metrics cache
to ensure each ratio test produces independent results.
"""

import subprocess
import json
import csv
import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple


def test_ratio(protected_ratio: float, dst_ratio: float, src_ratio: float) -> Dict:
    """Test a specific partition ratio."""
    
    # Sanity check
    total = protected_ratio + dst_ratio + src_ratio
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
    
    ratio_name = f"{protected_ratio:.2f}_{dst_ratio:.2f}_{src_ratio:.2f}"
    print(f"\n{'='*90}")
    print(f"Testing Norm-guided with ratio: {protected_ratio:.2f} / {dst_ratio:.2f} / {src_ratio:.2f}")
    print(f"Compression: {src_ratio/dst_ratio:.1f}x (if merging all src to dst)")
    print(f"{'='*90}")

    # Clear metrics cache for this test to ensure fresh evaluation
    output_path = Path(f"tests/tests_result/norm_ratio_test_{ratio_name}")
    print(f"🧹 Clearing metrics cache from: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"   ✓ Removed existing results")
    
    try:
        # Run evaluation
        cmd = [
            "python", "eval/eval_scannet.py",
            "--data_dir", "/home/hba/Documents/Dataset/ScanNet/scans",
            "--ckpt_path", "ckpt/model_tracker_fixed_e20.pt",
            "--input_frame", "20",
            "--num_scenes", "2",
            "--merging", "0",
            "--merge_ratio", "0.7",
            "--use_norm_guided",
            "--norm_protected_ratio", str(protected_ratio),
            "--norm_dst_ratio", str(dst_ratio),
            "--no_cache",  # Ensure fresh evaluation
            "--output_path", str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Parse metrics from output
            metrics = {}
            for line in output.split('\n'):
                if 'chamfer_distance:' in line:
                    metrics['cd'] = float(line.split(':')[1].strip())
                elif 'ate:' in line:
                    metrics['ate'] = float(line.split(':')[1].strip())
                elif 'are:' in line:
                    metrics['are'] = float(line.split(':')[1].strip())
                elif 'inference_time_ms:' in line:
                    metrics['time_ms'] = float(line.split(':')[1].strip())
            
            print(f"\n✓ Results:")
            print(f"  CD:  {metrics.get('cd', -1):.4f} cm")
            print(f"  ATE: {metrics.get('ate', -1):.4f} m")
            print(f"  ARE: {metrics.get('are', -1):.4f}°")
            print(f"  Time: {metrics.get('time_ms', -1):.0f} ms")
            
            return {
                'ratio': ratio_name,
                'protected': protected_ratio,
                'dst': dst_ratio,
                'src': src_ratio,
                'compression': src_ratio / dst_ratio if dst_ratio > 0 else 0,
                'cd': metrics.get('cd', None),
                'ate': metrics.get('ate', None),
                'are': metrics.get('are', None),
                'time_ms': metrics.get('time_ms', None),
                'success': True
            }
        else:
            print(f"✗ Evaluation failed")
            print(result.stderr)
            return {
                'ratio': ratio_name,
                'protected': protected_ratio,
                'dst': dst_ratio,
                'src': src_ratio,
                'compression': src_ratio / dst_ratio if dst_ratio > 0 else 0,
                'success': False,
                'error': result.stderr[:200]
            }
    
    finally:
        # Do not keep per-ratio outputs to avoid any cache reuse across runs
        if output_path.exists():
            shutil.rmtree(output_path)


def main():
    """Run optimization tests for different partition ratios."""
    
    print("\n" + "="*90)
    print("NORM-GUIDED PARTITION RATIO OPTIMIZATION TEST")
    print("="*90)
    
    # Test different ratios
    test_configs = [
        (0.10, 0.10, 0.80),  # Current (baseline)
        (0.10, 0.20, 0.70),  # Slightly more dst
        (0.10, 0.25, 0.65),  # Medium adjustment
        (0.10, 0.30, 0.60),  # Significant adjustment
        (0.10, 0.40, 0.50),  # Match Variance Top-K dst ratio
    ]
    
    results = []
    baseline_cd = None
    
    for protected, dst, src in test_configs:
        result = test_ratio(protected, dst, src)
        results.append(result)
        
        if protected == 0.10 and dst == 0.10:
            baseline_cd = result.get('cd')
    
    # Save results
    output_dir = Path("tests/tests_result/norm_partition_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / "partition_ratio_results.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['ratio', 'protected', 'dst', 'src', 'compression', 'cd', 'ate', 'are', 'time_ms', 'success', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    
    print(f"\n{'Ratio (P/D/S)':<15} {'Compression':<12} {'CD (cm)':<12} {'CD Change':<15} {'ATE (m)':<12} {'Time (ms)':<12}")
    print("-" * 90)
    
    for result in results:
        if result['success']:
            cd_change = ""
            if baseline_cd and result['cd']:
                pct_change = (result['cd'] - baseline_cd) / baseline_cd * 100
                cd_change = f"{pct_change:+.1f}%"
            
            print(f"{result['ratio']:<15} {result['compression']:>6.1f}x{'':<5} {result['cd']:>7.4f}{'':<3} {cd_change:<15} {result['ate']:>7.4f}{'':<3} {result['time_ms']:>8.0f}{'':<3}")
    
    print("\n✓ Results saved to:", csv_path)
    
    # Recommendations
    print("\n" + "="*90)
    print("RECOMMENDATIONS")
    print("="*90)
    
    valid_results = [r for r in results if r['success']]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x.get('cd', float('inf')))
        print(f"\n🏆 Best ratio for geometry: {best_result['ratio']} (CD: {best_result['cd']:.4f})")
        print(f"   Protected: {best_result['protected']:.1%}, Dst: {best_result['dst']:.1%}, Src: {best_result['src']:.1%}")
        print(f"   Compression: {best_result['compression']:.1f}x")
        
        # Compare with baseline
        if best_result['cd'] != baseline_cd:
            improvement = (baseline_cd - best_result['cd']) / baseline_cd * 100
            print(f"   Improvement over current: {improvement:+.1f}% CD")


if __name__ == "__main__":
    main()
