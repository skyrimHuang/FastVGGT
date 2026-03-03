"""
改进版实验一：基于L2范数的几何一致性Token划分有效性验证 (V2)
=================================================================

修复：参数链路直传
- 通过CLI参数将protected/dst比例传递到模型，不依赖环境变量
"""

import subprocess
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List
import time


def run_l2_partition_test_v2(
    frame_count: int,
    merge_ratio: float = 0.30,
    use_norm_guided: bool = False,
    protected_ratio: float = 0.10,
    dst_ratio: float = 0.40,
    test_name: str = "baseline"
) -> Dict:
    """
    改进版：运行Token划分方法测试，确保环境变量正确传递。
    """
    
    method_name = "L2范数指导" if use_norm_guided else "固定步长采样"
    print(f"\n{'='*90}")
    print(f"测试 {method_name}")
    print(f"  帧数: {frame_count} | 合并率: {merge_ratio:.1%}")
    if use_norm_guided:
        print(f"  保护比: {protected_ratio:.1%} | 目标比: {dst_ratio:.1%}")
    print(f"{'='*90}")
    
    # 创建输出目录
    output_subdir = f"l2_partition_test_{test_name}_f{frame_count}"
    output_path = Path(f"tests/tests_result/l2_partition_effectiveness_v2/{output_subdir}")
    
    # 清除之前的结果
    if output_path.exists():
        shutil.rmtree(output_path)
    
    start_time = time.time()
    
    try:
        # 构建eval_scannet.py的命令
        cmd = [
            sys.executable, "eval/eval_scannet.py",
            "--data_dir", "/home/hba/Documents/Dataset/ScanNet/scans",
            "--ckpt_path", "ckpt/model_tracker_fixed_e20.pt",
            "--input_frame", str(frame_count),
            "--num_scenes", "2",
            "--merging", "0",
            "--merge_ratio", str(merge_ratio),
            "--output_path", str(output_path),
            "--no_cache",
        ]
        
        # 添加方法特定的参数
        if use_norm_guided:
            cmd.extend([
                "--use_norm_guided",
                "--norm_protected_ratio", str(protected_ratio),
                "--norm_dst_ratio", str(dst_ratio),
            ])
        
        if use_norm_guided:
            print(f"  设置L2比例参数:")
            print(f"    --norm_protected_ratio={protected_ratio}")
            print(f"    --norm_dst_ratio={dst_ratio}")
        
        # 运行评估
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600, 
            cwd="/home/hba/Documents/FastVGGT",
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            output = result.stdout
            
            # 解析输出的指标
            metrics = {
                'cd': None,
                'ate': None,
                'are': None,
                'time_ms': None,
            }
            
            for line in output.split('\n'):
                if 'chamfer_distance:' in line:
                    metrics['cd'] = float(line.split(':')[1].strip())
                elif 'ate:' in line:
                    metrics['ate'] = float(line.split(':')[1].strip())
                elif 'are:' in line:
                    metrics['are'] = float(line.split(':')[1].strip())
                elif 'inference_time_ms:' in line:
                    metrics['time_ms'] = float(line.split(':')[1].strip())
            
            print(f"\n  ✓ 评估完成 (耗时: {elapsed_time:.1f}s)")
            print(f"    CD:  {metrics.get('cd', -1):.4f} cm")
            print(f"    ATE: {metrics.get('ate', -1):.4f} m")
            print(f"    ARE: {metrics.get('are', -1):.4f}°")
            print(f"    Time: {metrics.get('time_ms', -1):.0f} ms")
            
            return {
                'frame_count': frame_count,
                'method': method_name,
                'merge_ratio': merge_ratio,
                'protected_ratio': protected_ratio,
                'dst_ratio': dst_ratio,
                'cd': metrics.get('cd', None),
                'ate': metrics.get('ate', None),
                'are': metrics.get('are', None),
                'time_ms': metrics.get('time_ms', None),
                'elapsed_s': elapsed_time,
                'success': True
            }
        else:
            print(f"  ✗ 评估失败")
            print(f"  错误信息: {result.stderr[:300]}")
            
            return {
                'frame_count': frame_count,
                'method': method_name,
                'merge_ratio': merge_ratio,
                'protected_ratio': protected_ratio,
                'dst_ratio': dst_ratio,
                'success': False,
                'error': result.stderr[:200],
                'elapsed_s': elapsed_time
            }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ✗ 异常: {str(e)[:100]}")
        
        return {
            'frame_count': frame_count,
            'method': method_name,
            'merge_ratio': merge_ratio,
            'protected_ratio': protected_ratio,
            'dst_ratio': dst_ratio,
            'success': False,
            'error': str(e)[:100],
            'elapsed_s': elapsed_time
        }
    
    finally:
        # 清除输出目录以节省空间
        if output_path.exists():
            shutil.rmtree(output_path)


def main():
    """运行改进版的L2范数有效性验证实验。"""
    
    print("\n" + "="*90)
    print("实验一（改进版V2）：基于L2范数的几何一致性Token划分有效性验证")
    print("关键改进：正确传递环境变量到subprocess")
    print("="*90)
    
    # 实验参数
    frame_counts = [20,50,100,150]
    merge_ratio = 0.90
    
    # L2范数配置：使用推荐配置
    protected_ratio = 0.2
    dst_ratio = 0.6
    
    results = []
    
    # 对每个帧数进行对比测试
    for frame_count in frame_counts:
        print(f"\n\n{'*'*90}")
        print(f"测试序列长度: {frame_count} 帧")
        print(f"{'*'*90}")
        
        # 方法1：固定步长采样（基线）
        result_baseline = run_l2_partition_test_v2(
            frame_count=frame_count,
            merge_ratio=merge_ratio,
            use_norm_guided=False,
            test_name="baseline"
        )
        results.append(result_baseline)
        
        # 方法2：L2范数指导划分（本文方法）- 使用推荐参数
        result_l2 = run_l2_partition_test_v2(
            frame_count=frame_count,
            merge_ratio=merge_ratio,
            use_norm_guided=True,
            protected_ratio=protected_ratio,
            dst_ratio=dst_ratio,
            test_name="l2_norm"
        )
        results.append(result_l2)
    
    # 保存结果到CSV
    output_dir = Path("tests/tests_result/l2_partition_effectiveness_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "partition_comparison_results_v2.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['frame_count', 'method', 'merge_ratio', 'protected_ratio', 'dst_ratio', 
                     'cd', 'ate', 'are', 'time_ms', 'elapsed_s', 'success', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ 结果已保存: {csv_path}")
    
    # 生成对比总结
    print("\n" + "="*90)
    print("改进版实验结果总结")
    print("="*90)
    
    baseline_results = [r for r in results if r['success'] and not r['method'].startswith('L2')]
    l2_results = [r for r in results if r['success'] and r['method'].startswith('L2')]
    
    print(f"\n{'帧数':<8} {'方法':<15} {'保护%':<8} {'目标%':<8} {'CD(cm)':<12} {'改进':<12}")
    print("-" * 90)
    
    for baseline in baseline_results:
        print(f"{baseline['frame_count']:<8} {baseline['method']:<15} {'N/A':<8} {'N/A':<8} "
              f"{baseline['cd']:>7.4f}{'':<3} {'基线':>7}")
        
        l2_result = next((r for r in l2_results if r['frame_count'] == baseline['frame_count']), None)
        if l2_result and baseline['cd'] and baseline['cd'] > 0:  # 🔧 FIX: 检查baseline['cd']有效性
            cd_improvement = (baseline['cd'] - l2_result['cd']) / baseline['cd'] * 100
            print(f"{l2_result['frame_count']:<8} {l2_result['method']:<15} "
                  f"{l2_result['protected_ratio']:.0%}{'':<6} {l2_result['dst_ratio']:.0%}{'':<6} "
                  f"{l2_result['cd']:>7.4f}{'':<3} {cd_improvement:>+6.1f}%")
            print("-" * 90)
    
    # 关键发现
    print("\n" + "="*90)
    print("关键发现")
    print("="*90)
    
    if baseline_results and l2_results:
        all_improvements = []
        for baseline in baseline_results:
            l2_result = next((r for r in l2_results if r['frame_count'] == baseline['frame_count']), None)
            if l2_result and l2_result['cd']:
                    if baseline['cd'] and baseline['cd'] > 0:  # 🔧 FIX: 检查baseline['cd']有效性
                        improvement = (baseline['cd'] - l2_result['cd']) / baseline['cd'] * 100
                    else:
                        improvement = None
                    if improvement is not None:
                        all_improvements.append(improvement)
                        print(f"\n{baseline['frame_count']}帧: CD {'改进' if improvement > 0 else '恶化'} {abs(improvement):.2f}%")
                    else:
                        print(f"\n{baseline['frame_count']}帧: 数据无效（baseline CD为0）")
        
        if all_improvements:
            avg_improvement = sum(all_improvements) / len(all_improvements)
            print(f"\n  平均变化: {avg_improvement:+.2f}% {'(改进)' if avg_improvement > 0 else '(恶化)'}")
            print(f"\n【注意】：")
            if avg_improvement < 0:
                print(f"  如果CD变化为负，说明L2范数方法性能更差。")
                print(f"  可能原因：")
                print(f"  1. 环境变量在进程间传递有问题")
                print(f"  2. L2范数度量方式不适合当前任务")
                print(f"  3. 参数配置(P={protected_ratio:.0%}, D={dst_ratio:.0%})不是最优的")
            else:
                print(f"  L2范数方法成功改进了几何精度！")
    
    print("\n" + "="*90)


if __name__ == "__main__":
    main()
