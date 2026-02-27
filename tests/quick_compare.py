#!/usr/bin/env python3
"""
Quick comparison script to compare Norm-Only with Baseline
"""

import json
from pathlib import Path


def load_metrics(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_metrics():
    """Compare Norm-Only with Baseline"""
    baseline_path = Path("tests/tests_result/ablation/baseline/average_metrics.json")
    norm_only_path = Path("tests/tests_result/ablation/norm_only/input_frame_100/average_metrics.json")
    
    if not baseline_path.exists():
        print(f"❌ Baseline metrics not found at: {baseline_path}")
        return
    
    if not norm_only_path.exists():
        print(f"⏳ Norm-Only metrics not yet available at: {norm_only_path}")
        print(f"   Experiment may still be running...")
        return
    
    baseline = load_metrics(baseline_path)
    norm_only = load_metrics(norm_only_path)
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS: Norm-Guided vs Baseline")
    print("="*80 + "\n")
    
    metrics = [
        ('chamfer_distance', 'Chamfer Distance', True),
        ('ate', 'ATE (m)', True),
        ('are', 'ARE (deg)', True),
        ('inference_time_ms', 'Time (ms)', True)
    ]
    
    for metric_key, metric_name, lower_is_better in metrics:
        if metric_key not in baseline or metric_key not in norm_only:
            continue
        
        base_val = baseline[metric_key]
        norm_val = norm_only[metric_key]
        
        improvement = (base_val - norm_val) / base_val * 100
        
        if lower_is_better:
            better = norm_val < base_val
            symbol = '✓' if better else '✗'
        else:
            better = norm_val > base_val
            symbol = '✓' if better else '✗' 
        
        arrow = '↓' if improvement > 0 else '↑'
        color_code = '\033[92m' if better else '\033[91m'  # Green or red
        reset_code = '\033[0m'
        
        print(f"{metric_name:20s}: {base_val:10.4f} → {norm_val:10.4f}  "
              f"{color_code}({arrow} {abs(improvement):6.2f}%) {symbol}{reset_code}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    compare_metrics()
