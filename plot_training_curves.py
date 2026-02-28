#!/usr/bin/env python3
"""
绘制训练曲线 - Plot training curves from training_history.json

Usage:
    python plot_training_curves.py --history training_history.json --output curves.png
    python plot_training_curves.py --history outputs/scale_head/training_history.json
    python plot_training_curves.py --csv outputs/scale_head/training_history.csv
"""

import json
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_history_from_json(json_path: str) -> Dict[str, List]:
    """从JSON文件加载训练历史"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_history_from_csv(csv_path: str) -> Dict[str, List]:
    """从CSV文件加载训练历史"""
    history = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in history:
                    history[key] = []
                try:
                    # 尝试转换为float
                    history[key].append(float(value))
                except (ValueError, TypeError):
                    history[key].append(value)
    
    return history


def plot_training_curves(history: Dict[str, List], output_path: str = 'training_curves.png'):
    """绘制训练曲线"""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting")
        return False
    
    # 只绘制数值指标
    numeric_metrics = {}
    for key, values in history.items():
        if key in ['epoch', 'train_loss', 'val_mean_error', 'val_median_error', 
                   'val_rmse', 'val_mae', 'val_fps', 'val_inference_time', 'learning_rate']:
            numeric_metrics[key] = values
    
    if 'epoch' not in history:
        print("Warning: 'epoch' not found in history")
        return False
    
    epochs = history['epoch']
    num_epochs = len(epochs)
    
    # 根据可用的指标动态创建子图
    figures_created = []
    
    # 图1: 损失曲线
    if 'train_loss' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        loss_path = str(Path(output_path).parent / 'loss_curve.png')
        fig.savefig(loss_path, dpi=150, bbox_inches='tight')
        figures_created.append(loss_path)
        print(f"✓ Loss curve saved to: {loss_path}")
        plt.close(fig)
    
    # 图2: 验证误差曲线
    if 'val_mean_error' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, np.array(history['val_mean_error'])*100, 'g-s', label='Mean Error', linewidth=2, markersize=4)
        if 'val_median_error' in history:
            ax.plot(epochs, np.array(history['val_median_error'])*100, 'purple', marker='s', label='Median Error', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error (%)', fontsize=12)
        ax.set_title('Validation Error Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        error_path = str(Path(output_path).parent / 'error_curve.png')
        fig.savefig(error_path, dpi=150, bbox_inches='tight')
        figures_created.append(error_path)
        print(f"✓ Error curve saved to: {error_path}")
        plt.close(fig)
    
    # 图3: RMSE和MAE
    if 'val_rmse' in history or 'val_mae' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'val_rmse' in history:
            ax.plot(epochs, history['val_rmse'], 'r-^', label='RMSE', linewidth=2, markersize=4)
        if 'val_mae' in history:
            ax.plot(epochs, history['val_mae'], 'orange', marker='^', label='MAE', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error Value', fontsize=12)
        ax.set_title('RMSE and MAE Curves', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        metrics_path = str(Path(output_path).parent / 'rmse_mae_curve.png')
        fig.savefig(metrics_path, dpi=150, bbox_inches='tight')
        figures_created.append(metrics_path)
        print(f"✓ RMSE/MAE curve saved to: {metrics_path}")
        plt.close(fig)
    
    # 图4: FPS和推理时间
    if 'val_fps' in history or 'val_inference_time' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'val_fps' in history:
            ax.plot(epochs, history['val_fps'], 'cyan', marker='D', label='FPS', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('Inference Performance Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 添加推理时间到右侧y轴
        if 'val_inference_time' in history:
            ax2 = ax.twinx()
            ax2.plot(epochs, history['val_inference_time'], 'yellow', marker='D', label='Inference Time (ms)', linewidth=2, markersize=4)
            ax2.set_ylabel('Inference Time (ms)', fontsize=12)
        
        perf_path = str(Path(output_path).parent / 'performance_curve.png')
        fig.savefig(perf_path, dpi=150, bbox_inches='tight')
        figures_created.append(perf_path)
        print(f"✓ Performance curve saved to: {perf_path}")
        plt.close(fig)
    
    # 图5: 学习率变化
    if 'learning_rate' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(epochs, history['learning_rate'], 'purple', marker='o', label='Learning Rate', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        lr_path = str(Path(output_path).parent / 'learning_rate_curve.png')
        fig.savefig(lr_path, dpi=150, bbox_inches='tight')
        figures_created.append(lr_path)
        print(f"✓ Learning rate curve saved to: {lr_path}")
        plt.close(fig)
    
    # 图6: 组合曲线 (损失 + 验证误差)
    if 'train_loss' in history and 'val_mean_error' in history:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左y轴: 训练损失
        color = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Train Loss', color=color, fontsize=12)
        line1 = ax1.plot(epochs, history['train_loss'], color=color, marker='o', label='Train Loss', linewidth=2, markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # 右y轴: 验证误差
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Val Error (%)', color=color, fontsize=12)
        line2 = ax2.plot(epochs, np.array(history['val_mean_error'])*100, color=color, marker='s', label='Val Error', linewidth=2, markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 合并legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=11)
        
        fig.suptitle('Training Loss vs Validation Error', fontsize=14, fontweight='bold')
        combined_path = str(Path(output_path).parent / 'combined_curve.png')
        fig.savefig(combined_path, dpi=150, bbox_inches='tight')
        figures_created.append(combined_path)
        print(f"✓ Combined curve saved to: {combined_path}")
        plt.close(fig)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Training Statistics")
    print("="*60)
    if 'train_loss' in history:
        print(f"Train Loss:        min={min(history['train_loss']):.6f}, final={history['train_loss'][-1]:.6f}")
    if 'val_mean_error' in history:
        min_error = min(history['val_mean_error']) * 100
        final_error = history['val_mean_error'][-1] * 100
        print(f"Val Mean Error:    min={min_error:.2f}%, final={final_error:.2f}%")
    if 'val_rmse' in history:
        print(f"Val RMSE:          min={min(history['val_rmse']):.6f}, final={history['val_rmse'][-1]:.6f}")
    if 'val_fps' in history:
        print(f"Val FPS:           min={min(history['val_fps']):.2f}, max={max(history['val_fps']):.2f}")
    print(f"Total Epochs:      {num_epochs}")
    print("="*60)
    
    return len(figures_created) > 0


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from training_history.json or .csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_training_curves.py --history training_history.json
  python plot_training_curves.py --csv outputs/scale_head/training_history.csv
  python plot_training_curves.py --history training_history.json --output plots/
""")
    
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--csv', type=str, help='Path to training_history.csv')
    parser.add_argument('--output', type=str, default='training_curves.png',
                       help='Output path for curves (default: training_curves.png)')
    
    args = parser.parse_args()
    
    if args.history:
        history_path = args.history
        if not Path(history_path).exists():
            print(f"Error: {history_path} not found")
            return False
        
        print(f"Loading history from: {history_path}")
        history = load_history_from_json(history_path)
        
    elif args.csv:
        csv_path = args.csv
        if not Path(csv_path).exists():
            print(f"Error: {csv_path} not found")
            return False
        
        print(f"Loading history from: {csv_path}")
        history = load_history_from_csv(csv_path)
    else:
        # 尝试找默认的输出文件
        default_path = Path('outputs/scale_head/training_history.json')
        if default_path.exists():
            print(f"Used default path: {default_path}")
            history = load_history_from_json(str(default_path))
        else:
            print("Error: Please provide --history or --csv")
            parser.print_help()
            return False
    
    if not history:
        print("Error: No data loaded")
        return False
    
    # 确保输出目录存在
    output_dir = Path(args.output).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPlotting curves...")
    success = plot_training_curves(history, args.output)
    
    if success:
        print(f"\n✓ All curves plotted successfully!")
        return True
    else:
        print(f"\n✗ Failed to plot curves")
        return False


if __name__ == '__main__':
    main()
