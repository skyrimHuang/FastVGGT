#!/usr/bin/env python3
"""
Ablation Study Script for FastVGGT Token Merging Optimizations

This script runs systematic ablation experiments to isolate the effect of:
- Norm-Guided Anchoring (Protection mechanism)
- Threshold-Gated Anti-Collapse (Similarity filtering)
- Combined approach

Each experiment is compared against the baseline to measure improvement.
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Experiment configurations
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (Grid-based)",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/baseline",
        "description": "Original grid-based split without optimizations"
    },
    "norm_only": {
        "name": "Test A: Norm-Guided Only",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--use_norm_guided",
            "--merge_threshold", "0.0",  # Disable threshold
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/norm_only",
        "description": "Only L2 norm-guided anchoring with protection"
    },
    "threshold_only_75": {
        "name": "Test B1: Threshold-Gated (0.75)",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--merge_threshold", "0.75",
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/threshold_75",
        "description": "Similarity threshold filtering at 0.75 (no norm-guided)"
    },
    "threshold_only_85": {
        "name": "Test B2: Threshold-Gated (0.85)",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--merge_threshold", "0.85",
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/threshold_85",
        "description": "Similarity threshold filtering at 0.85 (no norm-guided)"
    },
    "combined_50": {
        "name": "Test C1: Combined (thresh=0.50)",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--use_norm_guided",
            "--merge_threshold", "0.50",
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/combined_50",
        "description": "Norm-guided with lower threshold 0.50"
    },
    "combined_75": {
        "name": "Test C2: Combined (thresh=0.75)",
        "args": [
            "--merging", "0",
            "--merge_ratio", "0.9",
            "--use_norm_guided",
            "--merge_threshold", "0.75",
            "--input_frame", "100",
            "--num_scenes", "5",
        ],
        "output_dir": "tests/tests_result/ablation/combined_75",
        "description": "Norm-guided with medium threshold 0.75"
    }
}

# Evaluation script
EVAL_SCRIPT = "eval/eval_scannet.py"
BASE_DIR = Path("/home/hba/Documents/FastVGGT")
CONDA_ENV = "fastvggt"


def run_experiment(exp_name, config):
    """Run a single experiment and return results path"""
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", EVAL_SCRIPT
    ] + [str(arg) for arg in config['args']]
    
    # Set output directory
    output_dir = BASE_DIR / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add output directory to args if not already specified
    if "--output_path" not in config['args']:
        cmd.extend(["--output_path", str(output_dir)])
    
    # Log command
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run experiment
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✓ {config['name']} completed in {duration:.1f}s")
        
        # Save stdout/stderr
        log_file = output_dir / "run_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start: {start_time}\n")
            f.write(f"End: {end_time}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"\n{'='*80}\nSTDOUT:\n{'='*80}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*80}\nSTDERR:\n{'='*80}\n")
            f.write(result.stderr)
        
        return {
            'success': True,
            'duration': duration,
            'output_dir': str(output_dir),
            'log_file': str(log_file)
        }
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✗ {config['name']} failed after {duration:.1f}s")
        print(f"Error: {e}")
        
        # Save error log
        error_log = output_dir / "error_log.txt"
        with open(error_log, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"\n{'='*80}\nSTDOUT:\n{'='*80}\n")
            f.write(e.stdout if e.stdout else "")
            f.write(f"\n{'='*80}\nSTDERR:\n{'='*80}\n")
            f.write(e.stderr if e.stderr else "")
        
        return {
            'success': False,
            'duration': duration,
            'output_dir': str(output_dir),
            'error': str(e),
            'error_log': str(error_log)
        }


def load_metrics(output_dir):
    """Load metrics from experiment output directory"""
    metrics_file = Path(output_dir) / "average_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def compare_with_baseline(baseline_metrics, exp_metrics, exp_name):
    """Compare experiment metrics with baseline and compute improvement"""
    if baseline_metrics is None or exp_metrics is None:
        return None
    
    comparison = {
        'experiment': exp_name,
        'metrics': {}
    }
    
    for metric_name in ['chamfer_distance', 'ate', 'are', 'inference_time_ms']:
        if metric_name not in baseline_metrics or metric_name not in exp_metrics:
            continue
        
        baseline_val = baseline_metrics[metric_name]
        exp_val = exp_metrics[metric_name]
        
        # For error metrics (smaller is better)
        if metric_name in ['chamfer_distance', 'ate', 'are']:
            improvement = (baseline_val - exp_val) / baseline_val * 100
            better = exp_val < baseline_val
        # For time (smaller is better)
        elif metric_name == 'inference_time_ms':
            improvement = (baseline_val - exp_val) / baseline_val * 100
            better = exp_val < baseline_val
        else:
            improvement = (exp_val - baseline_val) / baseline_val * 100
            better = exp_val > baseline_val
        
        comparison['metrics'][metric_name] = {
            'baseline': baseline_val,
            'experiment': exp_val,
            'improvement_pct': improvement,
            'better': better,
            'symbol': '↓' if better else '↑'
        }
    
    return comparison


def print_comparison_table(comparisons):
    """Print formatted comparison table"""
    print(f"\n{'='*100}")
    print("ABLATION STUDY RESULTS - Comparison with Baseline")
    print(f"{'='*100}\n")
    
    # Header
    metric_names = {
        'chamfer_distance': 'Chamfer Distance',
        'ate': 'ATE (m)',
        'are': 'ARE (deg)',
        'inference_time_ms': 'Time (ms)'
    }
    
    for comp in comparisons:
        print(f"\n{comp['experiment'].upper()}")
        print("-" * 100)
        
        for metric_key, metric_display in metric_names.items():
            if metric_key not in comp['metrics']:
                continue
            
            m = comp['metrics'][metric_key]
            symbol = '✓' if m['better'] else '✗'
            arrow = m['symbol']
            color = '\033[92m' if m['better'] else '\033[91m'  # Green if better, red if worse
            reset = '\033[0m'
            
            print(f"{metric_display:20s}: "
                  f"{m['baseline']:12.4f} → {m['experiment']:12.4f} "
                  f"{color}({arrow} {abs(m['improvement_pct']):6.2f}%) {symbol}{reset}")


def main():
    """Main execution flow"""
    print(f"\n{'='*100}")
    print("FastVGGT Token Merging - Ablation Study")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    # Check if baseline exists
    baseline_dir = BASE_DIR / EXPERIMENTS['baseline']['output_dir']
    baseline_metrics_file = baseline_dir / "average_metrics.json"
    
    # Results storage
    results = {}
    
    # Run experiments
    for exp_name, config in EXPERIMENTS.items():
        # Skip baseline if it already exists (use existing results)
        if exp_name == "baseline" and baseline_metrics_file.exists():
            print(f"\n{'='*80}")
            print(f"Using existing baseline results from: {baseline_dir}")
            print(f"{'='*80}\n")
            results[exp_name] = {
                'success': True,
                'output_dir': str(baseline_dir),
                'reused': True
            }
        else:
            results[exp_name] = run_experiment(exp_name, config)
    
    # Load metrics and compare
    print(f"\n{'='*100}")
    print("Loading metrics and computing comparisons...")
    print(f"{'='*100}\n")
    
    baseline_metrics = load_metrics(results['baseline']['output_dir'])
    if baseline_metrics is None:
        print("ERROR: Could not load baseline metrics!")
        sys.exit(1)
    
    comparisons = []
    for exp_name in EXPERIMENTS.keys():
        if exp_name == 'baseline':
            continue  # Skip baseline itself
        
        if exp_name not in results or not results[exp_name]['success']:
            print(f"Skipping {exp_name} (failed)")
            continue
        
        exp_metrics = load_metrics(results[exp_name]['output_dir'])
        if exp_metrics is None:
            print(f"Warning: Could not load metrics for {exp_name}")
            continue
        
        comp = compare_with_baseline(baseline_metrics, exp_metrics, EXPERIMENTS[exp_name]['name'])
        if comp:
            comparisons.append(comp)
    
    # Print comparison table
    print_comparison_table(comparisons)
    
    # Save comprehensive results
    output_file = BASE_DIR / "tests/tests_result/ablation/ablation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
        'baseline_metrics': baseline_metrics,
        'comparisons': comparisons
    }
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"Ablation study complete!")
    print(f"Results saved to: {output_file}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
