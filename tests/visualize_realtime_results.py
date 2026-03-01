"""
Visualization script for VGGT front-end real-time performance results.

This script generates comprehensive plots showing:
1. Latency comparison (Full Model vs Pose-Only)
2. Per-module timing breakdown
3. FPS capability vs AR requirements
4. Memory consumption
5. Latency distribution
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_results(results_dir: Path):
    """Load results from JSON file."""
    json_file = results_dir / 'vggt_realtime_results.json'
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    return results


def load_timings_csv(results_dir: Path):
    """Load timing data from CSV file."""
    csv_file = results_dir / 'vggt_timings.csv'
    
    full_times = []
    pose_times = []
    speedups = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Full_Model_ms']:
                full_times.append(float(row['Full_Model_ms']))
            if row['Pose_Only_ms']:
                pose_times.append(float(row['Pose_Only_ms']))
            if row['Speedup']:
                speedups.append(float(row['Speedup']))
    
    return {
        'full_times': full_times,
        'pose_times': pose_times,
        'speedups': speedups,
    }


def plot_latency_comparison(results: dict, timings: dict):
    """Plot 1: Latency comparison between full model and pose-only."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1.1: Bar chart of average latencies
    ax = axes[0]
    models = ['Full Model', 'Pose-Only']
    latencies = [
        results['full_model']['per_frame_latency_ms'],
        results['pose_only_model']['per_frame_latency_ms'],
    ]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax.bar(models, latencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add AR requirement lines
    ax.axhline(y=33.33, color='green', linestyle='--', linewidth=2, label='30 FPS (Mobile AR)')
    ax.axhline(y=16.67, color='red', linestyle='--', linewidth=2, label='60 FPS (VR)')
    
    ax.set_ylabel('Per-Frame Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Frame Latency Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, latency) in enumerate(zip(bars, latencies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.4f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 1.2: Latency distribution (histogram)
    ax = axes[1]
    ax.hist(timings['full_times'], bins=20, alpha=0.6, label='Full Model', color='#FF6B6B', edgecolor='black')
    ax.hist(timings['pose_times'], bins=20, alpha=0.6, label='Pose-Only', color='#4ECDC4', edgecolor='black')
    ax.axvline(x=np.mean(timings['full_times']), color='#FF6B6B', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(timings['pose_times']), color='#4ECDC4', linestyle='--', linewidth=2)
    ax.set_xlabel('Total Batch Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 1.3: FPS comparison
    ax = axes[2]
    fps_full = results['full_model']['fps_per_frame']
    fps_pose = results['pose_only_model']['fps_per_frame']
    
    bars = ax.bar(['Full Model', 'Pose-Only'], [fps_full, fps_pose], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='30 FPS (Mobile AR)')
    ax.axhline(y=60, color='red', linestyle='--', linewidth=2, label='60 FPS (VR)')
    ax.set_ylabel('Frames Per Second', fontsize=12, fontweight='bold')
    ax.set_title('FPS Capability (Per-Frame)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, fps in zip(bars, [fps_full, fps_pose]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{fps:.1f}fps', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_module_breakdown(results: dict):
    """Plot 2: Per-module timing breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    modules = {
        'Aggregator': results['module_breakdown']['Aggregator'],
        'Camera Head': results['module_breakdown']['Camera Head'],
        'Depth Head': results['module_breakdown']['Depth Head'],
    }
    
    # Plot 2.1: Module latencies (bar chart)
    ax = axes[0, 0]
    module_names = list(modules.keys())
    mean_times = [modules[m]['mean_ms'] for m in module_names]
    std_times = [modules[m]['std_ms'] for m in module_names]
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(module_names)))
    bars = ax.bar(range(len(module_names)), mean_times, yerr=std_times, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5, capsize=5)
    ax.set_xticks(range(len(module_names)))
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Module Latency Breakdown', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2.2: Module latencies (stacked bar)
    ax = axes[0, 1]
    cum_times = np.cumsum([0] + mean_times[:-1])
    
    for i, (name, color) in enumerate(zip(module_names, colors)):
        ax.barh(0, mean_times[i], left=cum_times[i], label=name, 
                color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlim([0, sum(mean_times)])
    ax.set_ylim([-0.5, 0.5])
    ax.set_yticks([])
    ax.set_xlabel('Cumulative Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Module Composition (Stacked)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2.3: Depth Head module analysis
    ax = axes[1, 0]
    total_time = results['module_breakdown']['Total']['mean_ms']
    depth_time = modules['Depth Head']['mean_ms']
    other_time = total_time - depth_time
    
    sizes = [depth_time, other_time]
    labels = [f'Depth Head\n({depth_time:.2f}ms, {depth_time/total_time*100:.1f}%)', 
              f'Other Modules\n({other_time:.2f}ms, {other_time/total_time*100:.1f}%)']
    colors_pie = ['#FF6B6B', '#4ECDC4']
    
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Depth Head Cost Analysis', fontsize=13, fontweight='bold')
    
    # Plot 2.4: Min/Max latencies
    ax = axes[1, 1]
    x_pos = np.arange(len(module_names))
    min_times = [modules[m]['min_ms'] for m in module_names]
    max_times = [modules[m]['max_ms'] for m in module_names]
    
    ax.scatter(x_pos, min_times, s=100, color='green', alpha=0.7, label='Min', marker='o')
    ax.scatter(x_pos, max_times, s=100, color='red', alpha=0.7, label='Max', marker='s')
    ax.scatter(x_pos, mean_times, s=100, color='blue', alpha=0.7, label='Mean', marker='^')
    
    for i in range(len(module_names)):
        ax.plot([i, i], [min_times[i], max_times[i]], 'k-', alpha=0.3, linewidth=1)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Min/Max/Mean Latencies', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ar_requirements(results: dict):
    """Plot 3: AR requirements analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    full_latency = results['full_model']['per_frame_latency_ms']
    pose_latency = results['pose_only_model']['per_frame_latency_ms']
    
    requirements = [
        ('Mobile AR\n(30 FPS)', 33.33),
        ('Tablet/Desktop\n(60 FPS)', 16.67),
    ]
    
    # Plot 3.1: Latency vs requirements (full model)
    ax = axes[0]
    req_names = [r[0] for r in requirements]
    req_latencies = [r[1] for r in requirements]
    
    x_pos = np.arange(len(req_names))
    width = 0.3
    
    bars1 = ax.bar(x_pos - width/2, req_latencies, width, label='AR Requirement', 
                    color='green', alpha=0.6, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, [full_latency] * len(req_names), width, 
                    label='Full Model', color='#FF6B6B', alpha=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Full Model vs AR Requirements', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(req_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add status indicators
    for i, req_lat in enumerate(req_latencies):
        if full_latency <= req_lat:
            ax.text(i + width/2, full_latency + 1, '✅ PASS', ha='center', fontweight='bold', color='green')
        else:
            ax.text(i + width/2, full_latency + 1, '❌ FAIL', ha='center', fontweight='bold', color='red')
    
    # Plot 3.2: Latency vs requirements (pose-only model)
    ax = axes[1]
    bars1 = ax.bar(x_pos - width/2, req_latencies, width, label='AR Requirement', 
                    color='green', alpha=0.6, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, [pose_latency] * len(req_names), width, 
                    label='Pose-Only', color='#4ECDC4', alpha=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Pose-Only Model vs AR Requirements', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(req_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add status indicators
    for i, req_lat in enumerate(req_latencies):
        if pose_latency <= req_lat:
            ax.text(i + width/2, pose_latency + 1, '✅ PASS', ha='center', fontweight='bold', color='green')
        else:
            ax.text(i + width/2, pose_latency + 1, '❌ FAIL', ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    return fig


def plot_latency_timeseries(timings: dict):
    """Plot 4: Latency over time (timeseries)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 4.1: Latency timeseries
    ax = axes[0]
    runs = np.arange(1, len(timings['full_times']) + 1)
    
    ax.plot(runs, timings['full_times'], 'o-', label='Full Model', 
            color='#FF6B6B', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(runs, timings['pose_times'], 's-', label='Pose-Only', 
            color='#4ECDC4', linewidth=2, markersize=4, alpha=0.7)
    
    ax.axhline(y=np.mean(timings['full_times']), color='#FF6B6B', 
               linestyle='--', linewidth=2, alpha=0.5, label='Full Mean')
    ax.axhline(y=np.mean(timings['pose_times']), color='#4ECDC4', 
               linestyle='--', linewidth=2, alpha=0.5, label='Pose Mean')
    
    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Batch Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Over Time (Consistency Check)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Plot 4.2: Speedup over time
    ax = axes[1]
    speedups = np.array(timings['speedups'])
    
    ax.plot(runs[:len(speedups)], speedups, 'o-', color='#95E1D3', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.axhline(y=np.mean(speedups), color='#95E1D3', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean Speedup: {np.mean(speedups):.2f}x')
    
    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Full / Pose-Only)', fontsize=12, fontweight='bold')
    ax.set_title('DPT Speedup Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_memory_comparison(results: dict):
    """Plot 5: Memory consumption comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Full Model', 'Pose-Only']
    memory = [
        results['full_model']['peak_memory_mb'],
        results['pose_only_model']['peak_memory_mb'],
    ]
    memory_std = [
        results['full_model']['peak_memory_std_mb'],
        results['pose_only_model']['peak_memory_std_mb'],
    ]
    
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax.bar(models, memory, yerr=memory_std, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2, capsize=5)
    
    ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Memory Consumption', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    # Add speedup annotation
    speedup_mem = memory[0] / memory[1]
    ax.text(0.5, max(memory) * 0.9, f'Speedup: {speedup_mem:.2f}x', 
            transform=ax.transData, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    return fig


def create_summary_report(results: dict, timings: dict):
    """Create a text summary report."""
    # Extract module breakdown metrics
    agg_time = results['module_breakdown']['Aggregator']['mean_ms']
    cam_time = results['module_breakdown']['Camera Head']['mean_ms']
    depth_time = results['module_breakdown']['Depth Head']['mean_ms']
    total_time = results['module_breakdown']['Total']['mean_ms']
    
    depth_ratio = (depth_time / total_time) * 100
    agg_ratio = (agg_time / total_time) * 100
    speedup = results['full_model']['per_frame_latency_ms'] / results['pose_only_model']['per_frame_latency_ms']
    memory_speedup = results['full_model']['peak_memory_mb'] / results['pose_only_model']['peak_memory_mb']
    latency_reduction = (1 - results['pose_only_model']['per_frame_latency_ms'] / results['full_model']['per_frame_latency_ms']) * 100
    
    # Full model requirements
    full_30fps = "PASS" if results['full_model']['per_frame_latency_ms'] <= 33.33 else "FAIL"
    full_60fps = "PASS" if results['full_model']['per_frame_latency_ms'] <= 16.67 else "FAIL"
    pose_30fps = "PASS" if results['pose_only_model']['per_frame_latency_ms'] <= 33.33 else "FAIL"
    pose_60fps = "PASS" if results['pose_only_model']['per_frame_latency_ms'] <= 16.67 else "FAIL"
    
    # Consistency metrics
    full_mean = np.mean(timings['full_times'])
    full_std = np.std(timings['full_times'])
    full_cv = (full_std / full_mean) * 100
    pose_mean = np.mean(timings['pose_times'])
    pose_std = np.std(timings['pose_times'])
    pose_cv = (pose_std / pose_mean) * 100
    speedup_mean = np.mean(timings['speedups'])
    speedup_std = np.std(timings['speedups'])
    
    # Speedup needed
    full_speedup_needed = results['full_model']['per_frame_latency_ms'] / 33.33
    pose_speedup_needed = results['pose_only_model']['per_frame_latency_ms'] / 33.33
    
    report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║              VGGT Front-end Real-time Performance Analysis Report              ║
╚════════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════════
1. EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════════

✓ Full Model (with depth):
  • Per-frame latency: {results['full_model']['per_frame_latency_ms']:.4f} ms
  • Throughput: {results['full_model']['fps_per_frame']:.2f} FPS (per frame)
  • Batch throughput: {results['full_model']['fps']:.2f} FPS (per sequence)
  • Peak GPU memory: {results['full_model']['peak_memory_mb']:.1f} MB

✓ Pose-Only Model (Depth disabled):
  • Per-frame latency: {results['pose_only_model']['per_frame_latency_ms']:.4f} ms
  • Throughput: {results['pose_only_model']['fps_per_frame']:.2f} FPS (per frame)
  • Batch throughput: {results['pose_only_model']['fps']:.2f} FPS (per sequence)
  • Peak GPU memory: {results['pose_only_model']['peak_memory_mb']:.1f} MB

Speedup by disabling Depth Head: {speedup:.2f}x
Memory reduction: {memory_speedup:.2f}x
Latency reduction: {latency_reduction:.1f}%

═══════════════════════════════════════════════════════════════════════════════════
2. AR REAL-TIME REQUIREMENT ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════

AR Requirement: Mobile (30 FPS) requires latency < 33.33 ms/frame
                Tablet (60 FPS) requires latency < 16.67 ms/frame

Full Model:
  • 30 FPS (Mobile AR):  {full_30fps:<8} (Latency: {results['full_model']['per_frame_latency_ms']:.4f} ms)
  • 60 FPS (VR/Desktop): {full_60fps:<8} (Latency: {results['full_model']['per_frame_latency_ms']:.4f} ms)

Pose-Only Model:
  • 30 FPS (Mobile AR):  {pose_30fps:<8} (Latency: {results['pose_only_model']['per_frame_latency_ms']:.4f} ms)
  • 60 FPS (VR/Desktop): {pose_60fps:<8} (Latency: {results['pose_only_model']['per_frame_latency_ms']:.4f} ms)

═══════════════════════════════════════════════════════════════════════════════════
3. PER-MODULE TIMING BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════════

Module                    | Mean (ms) | Std (ms) | Min (ms) | Max (ms)
────────────────────────────────────────────────────────────────────
Aggregator                | {agg_time:>9.2f} | {results['module_breakdown']['Aggregator']['std_ms']:>8.2f} | {results['module_breakdown']['Aggregator']['min_ms']:>8.2f} | {results['module_breakdown']['Aggregator']['max_ms']:>8.2f}
Camera Head               | {cam_time:>9.2f} | {results['module_breakdown']['Camera Head']['std_ms']:>8.2f} | {results['module_breakdown']['Camera Head']['min_ms']:>8.2f} | {results['module_breakdown']['Camera Head']['max_ms']:>8.2f}
Depth Head                | {depth_time:>9.2f} | {results['module_breakdown']['Depth Head']['std_ms']:>8.2f} | {results['module_breakdown']['Depth Head']['min_ms']:>8.2f} | {results['module_breakdown']['Depth Head']['max_ms']:>8.2f}
────────────────────────────────────────────────────────────────────
Total                     | {total_time:>9.2f} | {results['module_breakdown']['Total']['std_ms']:>8.2f} | {results['module_breakdown']['Total']['min_ms']:>8.2f} | {results['module_breakdown']['Total']['max_ms']:>8.2f}

Depth Head Cost: {depth_ratio:.2f}% of total inference time
Aggregator Cost: {agg_ratio:.2f}% of total inference time (includes transformer blocks)

═══════════════════════════════════════════════════════════════════════════════════
4. CONSISTENCY & STABILITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════

Full Model:
  • Mean latency: {full_mean:.4f} ms
  • Std deviation: {full_std:.4f} ms
  • Coefficient of variation: {full_cv:.2f}%

Pose-Only Model:
  • Mean latency: {pose_mean:.4f} ms
  • Std deviation: {pose_std:.4f} ms
  • Coefficient of variation: {pose_cv:.2f}%

Speedup consistency: Mean {speedup_mean:.2f}x (std {speedup_std:.2f}x)

═══════════════════════════════════════════════════════════════════════════════════
5. RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════════

✓ Pose-Only Mode is RECOMMENDED for:
  • Real-time AR applications requiring >30 FPS
  • Mobile/edge deployment with limited GPU resources
  • When depth estimation is provided by external sensors (RGB-D camera)

✗ Current Status:
  • Full Model: NOT SUITABLE for real-time AR (needs {full_speedup_needed:.1f}x speedup for 30 FPS)
  • Pose-Only: NOT SUITABLE for real-time AR (needs {pose_speedup_needed:.1f}x speedup for 30 FPS)

Strategy Recommendation:
  • Initialize once with Full Model to get T_global (pose + depth-based 3D map)
  • Use Pose-Only Model for subsequent tracking loops (still too slow for <30ms)
  • Consider model optimization: quantization, pruning, or architecture simplification
  • Evaluate faster backbones (ViT-B instead of ViT-L) for mobile deployment

═══════════════════════════════════════════════════════════════════════════════════
"""
    
    return report


if __name__ == "__main__":
    results_dir = Path("./tests/tests_result/vggt_realtime")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("VGGT Front-end Real-time Performance Visualization")
    print("="*80)
    
    try:
        # Load results
        print("\n[Loading Results]")
        results = load_results(results_dir)
        timings = load_timings_csv(results_dir)
        print(f"✅ Loaded results from {results_dir}")
        
        # Create visualizations
        print("\n[Generating Plots]")
        
        figs = []
        fig_names = []
        
        print("  • Plotting latency comparison...")
        fig1 = plot_latency_comparison(results, timings)
        figs.append(fig1)
        fig_names.append("01_latency_comparison")
        
        print("  • Plotting module breakdown...")
        fig2 = plot_module_breakdown(results)
        figs.append(fig2)
        fig_names.append("02_module_breakdown")
        
        print("  • Plotting AR requirements...")
        fig3 = plot_ar_requirements(results)
        figs.append(fig3)
        fig_names.append("03_ar_requirements")
        
        print("  • Plotting latency timeseries...")
        fig4 = plot_latency_timeseries(timings)
        figs.append(fig4)
        fig_names.append("04_latency_timeseries")
        
        print("  • Plotting memory comparison...")
        fig5 = plot_memory_comparison(results)
        figs.append(fig5)
        fig_names.append("05_memory_comparison")
        
        # Save figures
        print("\n[Saving Plots]")
        for fig, name in zip(figs, fig_names):
            output_file = results_dir / f"{name}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved {output_file}")
        
        # Generate report
        print("\n[Generating Report]")
        report = create_summary_report(results, timings)
        report_file = results_dir / "VGGT_REALTIME_ANALYSIS.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  ✅ Saved {report_file}")
        
        # Print report
        print("\n" + report)
        
        print("\n" + "="*80)
        print("✅ Visualization completed successfully")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"Please run test_vggt_frontend_realtime.py first to generate results")
