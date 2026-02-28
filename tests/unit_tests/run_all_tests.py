#!/usr/bin/env python3
"""
运行所有单元测试的主脚本
"""

import sys
from pathlib import Path
import subprocess

# 测试文件列表
TEST_FILES = [
    "test_01_kitti_calib.py",
    "test_02_scale_head.py",
    "test_03_vggt_integration.py",
    "test_04_kitti_dataset.py",
    "test_05_training_components.py",
]

def main():
    """运行所有测试"""
    unit_tests_dir = Path(__file__).parent
    
    print("=" * 80)
    print("运行所有单元测试")
    print("=" * 80)
    print(f"\n测试目录: {unit_tests_dir}\n")
    
    results = {}
    
    for test_file in TEST_FILES:
        test_path = unit_tests_dir / test_file
        
        if not test_path.exists():
            print(f"⚠ 测试文件不存在: {test_file}")
            results[test_file] = "SKIP"
            continue
        
        print(f"\n{'=' * 80}")
        print(f"运行: {test_file}")
        print(f"{'=' * 80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(unit_tests_dir),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                results[test_file] = "PASS"
            else:
                results[test_file] = "FAIL"
                
        except Exception as e:
            print(f"\n✗ 运行测试时出错: {e}")
            results[test_file] = "ERROR"
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_file, status in results.items():
        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "ERROR": "⚠",
            "SKIP": "⊘"
        }[status]
        
        print(f"{status_symbol} {test_file}: {status}")
    
    # 统计
    total = len(results)
    passed = sum(1 for s in results.values() if s == "PASS")
    failed = sum(1 for s in results.values() if s == "FAIL")
    errors = sum(1 for s in results.values() if s == "ERROR")
    skipped = sum(1 for s in results.values() if s == "SKIP")
    
    print(f"\n总计: {total} 个测试")
    print(f"  通过: {passed}")
    print(f"  失败: {failed}")
    print(f"  错误: {errors}")
    print(f"  跳过: {skipped}")
    
    if passed == total:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print(f"\n✗ {failed + errors}个测试未通过!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
