#!/usr/bin/env python3
"""
Ablation Study Script for FastVGGT Token Merging Optimizations

This script runs systematic ablation experiments to isolate the effect of:
- Norm-Guided Anchoring (Protection mechanism)
- Threshold-Gated Anti-Collapse (Similarity filtering)
- Combined approach

Each experiment is compared against the baseline to measure improvement.

Usage:
    # Run with default settings (100 frames, 5 scenes)
    python tests/run_ablation_study.py
    
    # Run with custom frames/scenes
    python tests/run_ablation_study.py --frames 20 --scenes 3
    
    # Run specific experiments only
    python tests/run_ablation_study.py --experiments baseline norm_only
    
    # Skip baseline if already exists
    python tests/run_ablation_study.py --skip_if_exists
"""

import subprocess
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import sys


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FastVGGT ablation study')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames per scene (default: 100)')
    parser.add_argument('--scenes', type=int, default=5,
                       help='Number of scenes to test (default: 5)')
    parser.add_argument('--conda_env', type=str, default='fastvggt',
                       help='Conda environment name (default: fastvggt)')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--skip_if_exists', action='store_true',
                       help='Skip experiments if results already exist')
    parser.add_argument('--merge_ratio', type=float, default=0.9,
                       help='Token merge ratio (default: 0.9)')
    parser.add_argument('--data_dir', type=str,
                       default='/home/hba/Documents/Dataset/ScanNet/scans',
                       help='Path to ScanNet data directory')
    parser.add_argument('--ckpt_path', type=str,
                       default='/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output from experiments')
    return parser.parse_args()


def get_experiments(num_frames, num_scenes, merge_ratio=0.9):
    """Generate experiment configurations"""
    base_args = [
        "--merging", "0",
        "--merge_ratio", str(merge_ratio),
        "--input_frame", str(num_frames),
        "--num_scenes", str(num_scenes),
    ]
    
    return {
        "baseline": {
            "name": "Baseline (Grid-based Top-K)",
            "args": base_args[:],
            "output_dir": f"tests/tests_result/ablation_f{num_frames}/baseline",
            "description": "Original grid-based split with Top-K selection"
        },
        "norm_guided": {
            "name": "Norm-Guided Anchoring (Top-K)",
            "args": base_args[:] + [
                "--use_norm_guided",
            ],
            "output_dir": f"tests/tests_result/ablation_f{num_frames}/norm_guided",
            "description": "L2 norm-guided anchoring with Top-K selection"
        },
    }


# Evaluation script
EVAL_SCRIPT = "eval/eval_scannet.py"
BASE_DIR = Path("/home/hba/Documents/FastVGGT")


def run_experiment(exp_name, config, conda_env, verbose=False):
    """Run a single experiment and return results path"""
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
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
    if verbose:
        print("Running in verbose mode...\n")
    
    # Run experiment
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=not verbose,
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
            if not verbose:
                f.write(f"\n{'='*80}\nSTDOUT:\n{'='*80}\n")
                f.write(result.stdout if result.stdout else "")
                f.write(f"\n{'='*80}\nSTDERR:\n{'='*80}\n")
                f.write(result.stderr if result.stderr else "")
        
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
            if not verbose:
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
    args = parse_args()
    
    print(f"\n{'='*100}")
    print("FastVGGT Token Merging - Ablation Study")
    print(f"Configuration: {args.frames} frames, {args.scenes} scenes, merge ratio {args.merge_ratio}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    # Get experiment configurations
    EXPERIMENTS = get_experiments(args.frames, args.scenes, args.merge_ratio)
    
    # Filter experiments if specific ones are requested
    if args.experiments:
        filtered_experiments = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
        if not filtered_experiments:
            print(f"ERROR: No valid experiments found. Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        EXPERIMENTS = filtered_experiments
        print(f"Running selected experiments: {list(EXPERIMENTS.keys())}\n")
    
    # Check if baseline exists
    baseline_config = EXPERIMENTS.get('baseline')
    if baseline_config:
        baseline_dir = BASE_DIR / baseline_config['output_dir']
        baseline_metrics_file = baseline_dir / "average_metrics.json"
    else:
        baseline_metrics_file = None
    
    # Results storage
    results = {}
    
    # Run experiments
    for exp_name, config in EXPERIMENTS.items():
        # Skip experiment if results exist and --skip_if_exists is set
        output_dir = BASE_DIR / config['output_dir']
        metrics_file = output_dir / "average_metrics.json"
        
        if args.skip_if_exists and metrics_file.exists():
            print(f"\n{'='*80}")
            print(f"Skipping {config['name']} - results already exist at: {output_dir}")
            print(f"{'='*80}\n")
            results[exp_name] = {
                'success': True,
                'output_dir': str(output_dir),
                'reused': True
            }
        else:
            results[exp_name] = run_experiment(exp_name, config, args.conda_env, args.verbose)
    
    # Load metrics and compare
    print(f"\n{'='*100}")
    print("Loading metrics and computing comparisons...")
    print(f"{'='*100}\n")
    
    # Find baseline (either from experiments or load existing)
    baseline_metrics = None
    if 'baseline' in results:
        baseline_metrics = load_metrics(results['baseline']['output_dir'])
    elif not args.experiments:
        # Try to find existing baseline
        baseline_dir = BASE_DIR / f"tests/tests_result/ablation_f{args.frames}/baseline"
        if baseline_dir.exists():
            baseline_metrics = load_metrics(str(baseline_dir))
            print(f"Using existing baseline from: {baseline_dir}\n")
    
    if baseline_metrics is None:
        print("WARNING: Could not load baseline metrics! Comparisons will be skipped.")
        comparisons = []
    else:
        comparisons = []
        for exp_name in EXPERIMENTS.keys():
            if exp_name == 'baseline':
                continue  # Skip baseline itself
            
            if exp_name not in results or not results[exp_name]['success']:
                print(f"Skipping {exp_name} (failed or not run)")
                continue
            
            exp_metrics = load_metrics(results[exp_name]['output_dir'])
            if exp_metrics is None:
                print(f"Warning: Could not load metrics for {exp_name}")
                continue
            
            comp = compare_with_baseline(baseline_metrics, exp_metrics, EXPERIMENTS[exp_name]['name'])
            if comp:
                comparisons.append(comp)
    
    # Print comparison table
    if comparisons:
        print_comparison_table(comparisons)
    
    # Save comprehensive results
    output_file = BASE_DIR / f"tests/tests_result/ablation_f{args.frames}/ablation_results_{args.scenes}scenes.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'frames': args.frames,
            'scenes': args.scenes,
            'merge_ratio': args.merge_ratio,
        },
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
    
    # Print summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"Summary: {successful}/{total} experiments completed successfully")
    if successful < total:
        failed = [name for name, r in results.items() if not r['success']]
        print(f"Failed experiments: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    main()
