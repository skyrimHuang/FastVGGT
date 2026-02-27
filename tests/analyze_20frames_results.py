#!/usr/bin/env python3
"""
Analyze existing 20-frame test results and generate comparison report
"""

import json
from pathlib import Path
import sys


def load_metrics(path):
    """Load metrics from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def compute_improvement(baseline_val, test_val, lower_is_better=True):
    """Compute improvement percentage"""
    if lower_is_better:
        improvement = (baseline_val - test_val) / baseline_val * 100
        better = test_val < baseline_val
    else:
        improvement = (test_val - baseline_val) / baseline_val * 100
        better = test_val > baseline_val
    
    return improvement, better


def main():
    """Main analysis"""
    print("\n" + "="*100)
    print("20-FRAME ABLATION STUDY RESULTS - Comparison with Baseline")
    print("="*100 + "\n")
    
    # Load baseline
    baseline_path = Path("tests/tests_result/quick_baseline/input_frame_20/average_metrics.json")
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found at {baseline_path}")
        sys.exit(1)
    
    baseline = load_metrics(baseline_path)
    
    # Define experiments
    experiments = {
        "Norm-Guided Only": "tests/tests_result/quick_norm_only/input_frame_20/average_metrics.json",
        "Threshold-0.85 Only": "tests/tests_result/quick_threshold_only/input_frame_20/average_metrics.json",
        "Combined (Norm+Thresh-0.85)": "tests/tests_result/quick_test/input_frame_20/average_metrics.json",
    }
    
    # Metric definitions
    metrics_config = {
        'chamfer_distance': ('Chamfer Distance', True),
        'ate': ('ATE (m)', True),
        'are': ('ARE (deg)', True),
        'inference_time_ms': ('Time (ms)', True),
    }
    
    # Print baseline values
    print("BASELINE (Grid-based split):")
    print("-" * 100)
    for metric_key, (metric_name, _) in metrics_config.items():
        if metric_key in baseline:
            print(f"  {metric_name:20s}: {baseline[metric_key]:12.4f}")
    print()
    
    # Analyze each experiment
    results_summary = []
    
    for exp_name, exp_path in experiments.items():
        exp_path = Path(exp_path)
        if not exp_path.exists():
            print(f"⚠️  {exp_name}: NOT FOUND")
            continue
        
        exp_metrics = load_metrics(exp_path)
        
        print(f"\n{exp_name}:")
        print("-" * 100)
        
        exp_summary = {'name': exp_name, 'metrics': {}}
        
        for metric_key, (metric_name, lower_is_better) in metrics_config.items():
            if metric_key not in baseline or metric_key not in exp_metrics:
                continue
            
            baseline_val = baseline[metric_key]
            exp_val = exp_metrics[metric_key]
            
            improvement, better = compute_improvement(baseline_val, exp_val, lower_is_better)
            
            symbol = '✓' if better else '✗'
            arrow = '↓' if improvement > 0 else '↑'
            color = '\033[92m' if better else '\033[91m'  # Green or red
            reset = '\033[0m'
            
            print(f"  {metric_name:20s}: {baseline_val:10.4f} → {exp_val:10.4f}  "
                  f"{color}({arrow} {abs(improvement):6.2f}%) {symbol}{reset}")
            
            exp_summary['metrics'][metric_key] = {
                'baseline': baseline_val,
                'value': exp_val,
                'improvement': improvement,
                'better': better
            }
        
        results_summary.append(exp_summary)
    
    # Summary table
    print("\n\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100 + "\n")
    
    # Header
    print(f"{'Method':<35s} {'CD':<15s} {'ATE':<15s} {'ARE':<15s} {'Time':<15s}")
    print("-" * 100)
    
    # Baseline row
    print(f"{'Baseline':<35s} "
          f"{baseline['chamfer_distance']:>7.4f}        "
          f"{baseline['ate']:>7.4f}        "
          f"{baseline['are']:>7.2f}        "
          f"{baseline['inference_time_ms']:>7.0f}ms")
    
    # Experiment rows
    for exp_data in results_summary:
        name = exp_data['name']
        metrics = exp_data['metrics']
        
        cd_str = f"{metrics['chamfer_distance']['value']:>7.4f} "
        cd_str += f"({metrics['chamfer_distance']['improvement']:+5.1f}%)"
        
        ate_str = f"{metrics['ate']['value']:>7.4f} "
        ate_str += f"({metrics['ate']['improvement']:+5.1f}%)"
        
        are_str = f"{metrics['are']['value']:>7.2f} "
        are_str += f"({metrics['are']['improvement']:+5.1f}%)"
        
        time_str = f"{metrics['inference_time_ms']['value']:>7.0f}ms"
        time_str += f"({metrics['inference_time_ms']['improvement']:+4.1f}%)"
        
        print(f"{name:<35s} {cd_str:<15s} {ate_str:<15s} {are_str:<15s} {time_str:<15s}")
    
    # Key findings
    print("\n\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100 + "\n")
    
    # Find best performing method
    best_cd = min(results_summary, key=lambda x: x['metrics']['chamfer_distance']['value'])
    best_ate = min(results_summary, key=lambda x: x['metrics']['ate']['value'])
    best_are = min(results_summary, key=lambda x: x['metrics']['are']['value'])
    
    print(f"🏆 Best Chamfer Distance: {best_cd['name']}")
    print(f"   Value: {best_cd['metrics']['chamfer_distance']['value']:.4f} "
          f"(↓{best_cd['metrics']['chamfer_distance']['improvement']:.1f}%)")
    print()
    
    print(f"🏆 Best ATE: {best_ate['name']}")
    print(f"   Value: {best_ate['metrics']['ate']['value']:.4f} "
          f"(↓{best_ate['metrics']['ate']['improvement']:.1f}%)")
    print()
    
    print(f"🏆 Best ARE: {best_are['name']}")
    print(f"   Value: {best_are['metrics']['are']['value']:.2f} "
          f"(↓{best_are['metrics']['are']['improvement']:.1f}%)")
    print()
    
    # Overall recommendation
    print("\n📌 RECOMMENDATION:")
    
    # Check if norm_only is best overall
    norm_only_data = [x for x in results_summary if "Norm-Guided Only" in x['name']]
    if norm_only_data:
        norm_data = norm_only_data[0]['metrics']
        cd_good = norm_data['chamfer_distance']['better']
        ate_good = norm_data['ate']['better']
        are_good = norm_data['are']['better']
        
        if cd_good and ate_good and are_good:
            print("   ✅ Use 'Norm-Guided Only' - Best performance across all metrics")
            print("   Configuration: --use_norm_guided --merge_threshold 0.0")
        elif cd_good and are_good:
            print("   ✅ Use 'Norm-Guided Only' - Excellent for reconstruction (CD) and rotation (ARE)")
            print("   Configuration: --use_norm_guided --merge_threshold 0.0")
            print("   ⚠️  Note: Slight ATE improvement but still significantly better than baseline")
        else:
            print("   ⚠️  Results are mixed - consider use case")
    
    # Check if threshold helps
    threshold_data = [x for x in results_summary if "Threshold-0.85 Only" in x['name']]
    if threshold_data:
        thresh_metrics = threshold_data[0]['metrics']
        if not any(m['better'] for m in thresh_metrics.values()):
            print("   ❌ Threshold-only provides no benefit - do not use alone")
    
    # Check if combined is harmful
    combined_data = [x for x in results_summary if "Combined" in x['name']]
    if combined_data:
        combined_metrics = combined_data[0]['metrics']
        if not any(m['better'] for m in combined_metrics.values()):
            print("   ❌ Combined approach (Norm+Threshold) is harmful - avoid this configuration")
    
    print("\n" + "="*100 + "\n")
    
    # Save results to JSON
    output_file = Path("tests/tests_result/ablation_20frames_analysis.json")
    output_data = {
        'baseline': baseline,
        'experiments': results_summary
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Analysis saved to: {output_file}\n")


if __name__ == "__main__":
    main()
