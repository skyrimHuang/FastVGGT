#!/usr/bin/env python3
"""
Configurable Ablation Study for FastVGGT Token Merging

Usage:
    python tests/run_ablation_configurable.py --frames 20 --scenes 5
    python tests/run_ablation_configurable.py --frames 50 --scenes 3
    python tests/run_ablation_configurable.py --frames 100 --scenes 2
"""

import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run configurable ablation study')
    parser.add_argument('--frames', type=int, default=20, 
                       help='Number of frames per scene (default: 20)')
    parser.add_argument('--scenes', type=int, default=5,
                       help='Number of scenes to test (default: 5)')
    parser.add_argument('--conda_env', type=str, default='fastvggt',
                       help='Conda environment name (default: fastvggt)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline test if already exists')
    return parser.parse_args()


# Experiment configurations
def get_experiments(num_frames, num_scenes):
    """Generate experiment configurations"""
    base_args = [
        "--merging", "0",
        "--merge_ratio", "0.9",
        "--input_frame", str(num_frames),
        "--num_scenes", str(num_scenes),
    ]
    
    return {
        "baseline": {
            "name": "Baseline (Grid-based)",
            "args": base_args[:],
            "output_dir": f"tests/tests_result/ablation_f{num_frames}/baseline",
            "description": "Original grid-based split (Top-K selection)"
        },
        "norm_guided": {
            "name": "Norm-Guided Anchoring",
            "args": base_args[:] + [
                "--use_norm_guided",
            ],
            "output_dir": f"tests/tests_result/ablation_f{num_frames}/norm_guided",
            "description": "L2 norm-guided anchoring with Top-K selection"
        },
    }


EVAL_SCRIPT = "eval/eval_scannet.py"
BASE_DIR = Path("/home/hba/Documents/FastVGGT")


def run_experiment(exp_name, config, conda_env):
    """Run a single experiment"""
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
    cmd.extend(["--output_path", str(output_dir)])
    
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
        
        return {
            'success': True,
            'duration': duration,
            'output_dir': str(output_dir)
        }
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✗ {config['name']} failed after {duration:.1f}s")
        print(f"Error: {e}")
        
        return {
            'success': False,
            'duration': duration,
            'output_dir': str(output_dir),
            'error': str(e)
        }


def load_metrics(output_dir, num_frames):
    """Load metrics from experiment output directory"""
    # Try multiple possible paths
    possible_paths = [
        Path(output_dir) / f"input_frame_{num_frames}" / "average_metrics.json",
        Path(output_dir) / "average_metrics.json",
    ]
    
    for metrics_file in possible_paths:
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
    
    # Try globbing
    import glob
    pattern = str(Path(output_dir) / "input_frame_*" / "average_metrics.json")
    matches = glob.glob(pattern)
    if matches:
        with open(matches[0], 'r') as f:
            return json.load(f)
    
    return None


def compare_with_baseline(baseline_metrics, exp_metrics, exp_name):
    """Compare experiment with baseline"""
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
        if metric_name in ['chamfer_distance', 'ate', 'are', 'inference_time_ms']:
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


def print_comparison_table(comparisons, num_frames):
    """Print formatted comparison table"""
    print(f"\n{'='*100}")
    print(f"ABLATION STUDY RESULTS (Frames={num_frames}) - Comparison with Baseline")
    print(f"{'='*100}\n")
    
    metric_names = {
        'chamfer_distance': 'Chamfer Distance',
        'ate': 'ATE (m)',
        'are': 'ARE (deg)',
        'inference_time_ms': 'Time (ms)'
    }
    
    for comp in comparisons:
        print(f"\n{comp['experiment']}")
        print("-" * 100)
        
        for metric_key, metric_display in metric_names.items():
            if metric_key not in comp['metrics']:
                continue
            
            m = comp['metrics'][metric_key]
            symbol = '✓' if m['better'] else '✗'
            arrow = m['symbol']
            color = '\033[92m' if m['better'] else '\033[91m'
            reset = '\033[0m'
            
            print(f"{metric_display:20s}: "
                  f"{m['baseline']:12.4f} → {m['experiment']:12.4f} "
                  f"{color}({arrow} {abs(m['improvement_pct']):6.2f}%) {symbol}{reset}")


def main():
    """Main execution"""
    args = parse_args()
    
    print(f"\n{'='*100}")
    print(f"FastVGGT Token Merging - Ablation Study")
    print(f"Frames: {args.frames}, Scenes: {args.scenes}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    # Get experiments
    experiments = get_experiments(args.frames, args.scenes)
    
    # Check if baseline exists
    baseline_dir = BASE_DIR / experiments['baseline']['output_dir']
    baseline_metrics_pattern = f"input_frame_{args.frames}/average_metrics.json"
    baseline_metrics_file = baseline_dir / baseline_metrics_pattern
    
    results = {}
    
    # Run experiments
    for exp_name, config in experiments.items():
        # Skip baseline if requested and exists
        if exp_name == "baseline" and args.skip_baseline and baseline_metrics_file.exists():
            print(f"\n{'='*80}")
            print(f"Using existing baseline results from: {baseline_dir}")
            print(f"{'='*80}\n")
            results[exp_name] = {
                'success': True,
                'output_dir': str(baseline_dir),
                'reused': True
            }
        else:
            results[exp_name] = run_experiment(exp_name, config, args.conda_env)
    
    # Load metrics and compare
    print(f"\n{'='*100}")
    print("Loading metrics and computing comparisons...")
    print(f"{'='*100}\n")
    
    baseline_metrics = load_metrics(results['baseline']['output_dir'], args.frames)
    if baseline_metrics is None:
        print("ERROR: Could not load baseline metrics!")
        sys.exit(1)
    
    comparisons = []
    for exp_name in experiments.keys():
        if exp_name == 'baseline':
            continue
        
        if exp_name not in results or not results[exp_name]['success']:
            print(f"Skipping {exp_name} (failed)")
            continue
        
        exp_metrics = load_metrics(results[exp_name]['output_dir'], args.frames)
        if exp_metrics is None:
            print(f"Warning: Could not load metrics for {exp_name}")
            continue
        
        comp = compare_with_baseline(baseline_metrics, exp_metrics, experiments[exp_name]['name'])
        if comp:
            comparisons.append(comp)
    
    # Print comparison
    print_comparison_table(comparisons, args.frames)
    
    # Save results
    output_file = BASE_DIR / f"tests/tests_result/ablation_f{args.frames}/comparison_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'num_frames': args.frames,
        'num_scenes': args.scenes,
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
