#!/usr/bin/env python3
"""
单元测试1: KITTI Calibration处理模块测试
测试calibration文件解析和参数缩放功能
"""

import sys
from pathlib import Path

# 添加项目根目录（tests/unit_tests/test_01... -> tests/unit_tests/ -> tests/ -> FastVGGT/）
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

def test_imports():
    """测试模块导入"""
    print("TEST 1: 测试模块导入...")
    try:
        from eval.dataset_utils.kitti_calib import (
            KITTICalibrationProcessor,
            load_kitti_calibration
        )
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_calibration_processor_class():
    """测试KITTICalibrationProcessor类"""
    print("\nTEST 2: 测试KITTICalibrationProcessor类结构...")
    try:
        from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor
        
        processor = KITTICalibrationProcessor()
        
        # 检查类属性
        assert hasattr(processor, 'KITTI_ORIGINAL_SIZE')
        assert hasattr(processor, 'VGGT_TARGET_SIZE')
        assert processor.KITTI_ORIGINAL_SIZE == (1242, 375)
        assert processor.VGGT_TARGET_SIZE == (518, 392)
        
        # 检查方法存在
        assert hasattr(processor, 'parse_calib_file')
        assert hasattr(processor, 'scale_intrinsics')
        assert hasattr(processor, 'compute_target_resolution')
        
        print("✓ 类结构正确")
        return True
    except Exception as e:
        print(f"✗ 类结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scale_intrinsics_math():
    """测试内参矩阵缩放的数学正确性"""
    print("\nTEST 3: 测试内参矩阵缩放...")
    try:
        from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor
        
        # 创建测试K矩阵
        K_orig = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.8540],
            [0.0, 0.0, 1.0]
        ])
        
        orig_size = (1242, 375)
        target_size = (518, 392)
        
        K_scaled = KITTICalibrationProcessor.scale_intrinsics(
            K_orig, orig_size, target_size
        )
        
        # 计算预期的缩放因子
        expected_scale_x = 518 / 1242  # ≈ 0.417
        expected_scale_y = 392 / 375   # ≈ 1.045
        
        # 验证焦距缩放
        assert abs(K_scaled[0, 0] / K_orig[0, 0] - expected_scale_x) < 1e-6
        assert abs(K_scaled[1, 1] / K_orig[1, 1] - expected_scale_y) < 1e-6
        
        # 验证主点缩放
        assert abs(K_scaled[0, 2] / K_orig[0, 2] - expected_scale_x) < 1e-6
        assert abs(K_scaled[1, 2] / K_orig[1, 2] - expected_scale_y) < 1e-6
        
        print(f"  原始焦距: fx={K_orig[0,0]:.2f}, fy={K_orig[1,1]:.2f}")
        print(f"  缩放焦距: fx={K_scaled[0,0]:.2f}, fy={K_scaled[1,1]:.2f}")
        print(f"  缩放因子: x={expected_scale_x:.4f}, y={expected_scale_y:.4f}")
        print("✓ 内参矩阵缩放正确")
        return True
    except Exception as e:
        print(f"✗ 内参矩阵缩放测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_target_resolution():
    """测试目标分辨率计算"""
    print("\nTEST 4: 测试目标分辨率计算...")
    try:
        from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor
        
        # 测试标准KITTI尺寸
        target_w, target_h = KITTICalibrationProcessor.compute_target_resolution(
            orig_w=1242, orig_h=375, target_w=518
        )
        
        # 验证是14的倍数
        assert target_w % 14 == 0, f"宽度{target_w}不是14的倍数"
        assert target_h % 14 == 0, f"高度{target_h}不是14的倍数"
        
        # 验证结果
        # scale = 518/1242 ≈ 0.417
        # target_h = ceil(375 * 0.417 / 14) * 14 = ceil(156.375 / 14) * 14 = ceil(11.17) * 14 = 168
        assert target_w == 518
        assert target_h == 168
        
        print(f"  输入: 1242×375")
        print(f"  输出: {target_w}×{target_h}")
        print(f"  验证: {target_w}%14={target_w%14}, {target_h}%14={target_h%14}")
        print("✓ 目标分辨率计算正确")
        return True
    except Exception as e:
        print(f"✗ 目标分辨率计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verify_calibration_scaling():
    """测试calibration缩放验证功能"""
    print("\nTEST 5: 测试缩放验证功能...")
    try:
        from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor
        
        K_orig = np.array([
            [700.0, 0.0, 600.0],
            [0.0, 700.0, 180.0],
            [0.0, 0.0, 1.0]
        ])
        
        scale_x, scale_y = 0.5, 1.0
        
        # 正确缩放
        K_scaled_correct = K_orig.copy()
        K_scaled_correct[0, 0] *= scale_x
        K_scaled_correct[1, 1] *= scale_y
        K_scaled_correct[0, 2] *= scale_x
        K_scaled_correct[1, 2] *= scale_y
        
        result = KITTICalibrationProcessor.verify_calibration_scaling(
            K_orig, K_scaled_correct, scale_x, scale_y
        )
        
        assert result == True, "正确缩放应该验证通过"
        
        # 错误缩放
        K_scaled_wrong = K_orig.copy()
        K_scaled_wrong[0, 0] *= (scale_x + 0.1)  # 故意错误
        
        result = KITTICalibrationProcessor.verify_calibration_scaling(
            K_orig, K_scaled_wrong, scale_x, scale_y
        )
        
        assert result == False, "错误缩放应该验证失败"
        
        print("✓ 缩放验证功能正确")
        return True
    except Exception as e:
        print(f"✗ 缩放验证功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("单元测试: KITTI Calibration处理模块")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_calibration_processor_class,
        test_scale_intrinsics_math,
        test_compute_target_resolution,
        test_verify_calibration_scaling,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print(f"\n✗ {total - passed}个测试失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
