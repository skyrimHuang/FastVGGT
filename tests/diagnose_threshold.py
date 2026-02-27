#!/usr/bin/env python3
"""
Diagnostic script to analyze why threshold causes negative impact when combined with norm-guided.
Outputs detailed statistics about token merging behavior.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from eval.data import get_scannet_test_data_loader
from eval.eval_custom import build_model
import yaml


def analyze_merge_behavior(model, data_loader, config_name):
    """Run inference and collect merge statistics"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {config_name}")
    print(f"{'='*80}\n")
    
    model.eval()
    stats = {
        'similarity_scores': [],
        'merge_ratios': [],
        'valid_count_ratios': [],
        'r_actual_vs_r': []
    }
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if batch_idx >= 5:  # Analyze first 5 batches
                break
            
            # Run model (this will trigger token merging)
            _ = model(data)
            
            # Note: We need to instrument the merge function to collect stats
            # For now, let's print what we can
            print(f"Batch {batch_idx + 1} processed")
    
    return stats


def test_threshold_values():
    """Test different threshold values to find optimal range"""
    print("\n" + "="*80)
    print("Testing different threshold values")
    print("="*80 + "\n")
    
    # Load model
    ckpt_path = "ckpt/model_tracker_fixed_e20.pt"
    
    # Test configurations
    thresholds = [0.0, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
    
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold:.2f}")
        
        # Create model with specific threshold
        # Note: This requires modifying model init to accept threshold
        # For now, we can test via command line
        
        import subprocess
        cmd = [
            "conda", "run", "-n", "fastvggt",
            "python", "eval/eval_scannet.py",
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--use_norm_guided",
            "--merge_threshold", str(threshold),
            "--input_frame", "10",  # Very short test
            "--num_scenes", "1",
            "--output_path", f"tests/tests_result/threshold_sweep/thresh_{threshold:.2f}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse output for metrics
            output = result.stdout
            if "ate:" in output:
                ate_line = [l for l in output.split('\n') if l.startswith('ate:')][0]
                ate_val = float(ate_line.split(':')[1].strip())
                
                are_line = [l for l in output.split('\n') if l.startswith('are:')][0]
                are_val = float(are_line.split(':')[1].strip())
                
                cd_line = [l for l in output.split('\n') if l.startswith('chamfer_distance:')][0]
                cd_val = float(cd_line.split(':')[1].strip())
                
                results.append({
                    'threshold': threshold,
                    'ate': ate_val,
                    'are': are_val,
                    'chamfer_distance': cd_val
                })
                
                print(f"  ATE: {ate_val:.4f}, ARE: {are_val:.4f}, CD: {cd_val:.4f}")
        
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Threshold Sweep Results")
    print(f"{'='*80}\n")
    print(f"{'Threshold':<12} {'ATE':<12} {'ARE':<12} {'CD':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['ate']:<12.4f} {r['are']:<12.4f} {r['chamfer_distance']:<12.4f}")
    
    # Find best
    if results:
        best_ate = min(results, key=lambda x: x['ate'])
        best_are = min(results, key=lambda x: x['are'])
        best_cd = min(results, key=lambda x: x['chamfer_distance'])
        
        print(f"\nBest ATE at threshold={best_ate['threshold']:.2f}: {best_ate['ate']:.4f}")
        print(f"Best ARE at threshold={best_are['threshold']:.2f}: {best_are['are']:.4f}")
        print(f"Best CD at threshold={best_cd['threshold']:.2f}: {best_cd['chamfer_distance']:.4f}")


def main():
    """Main diagnostic flow"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['threshold_sweep', 'analyze_merge'], default='threshold_sweep')
    args = parser.parse_args()
    
    if args.test == 'threshold_sweep':
        test_threshold_values()
    else:
        print("Merge analysis not yet implemented")


if __name__ == "__main__":
    main()
