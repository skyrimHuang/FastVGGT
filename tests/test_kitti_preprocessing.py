"""KITTI预处理验证脚本

验证数据加载、calibration缩放和scale标签计算的正确性
"""

import sys
from pathlib import Path

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from eval.dataset_utils.kitti_stereo import KITTIStereoDataset
from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor


def test_dataset_loading():
    """测试数据集加载"""
    print("=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    try:
        dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI")
        print(f"✓ Dataset loaded successfully")
        print(f"  - Dataset size: {len(dataset)} samples")
        print(f"  - Original resolution: {dataset.ORIGINAL_SIZE}")
        print(f"  - Target resolution: {dataset.TARGET_SIZE}")
        print()
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    return True


def test_single_sample():
    """测试单个样本加载"""
    print("=" * 60)
    print("TEST 2: Single Sample Loading")
    print("=" * 60)
    
    dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI")
    
    try:
        sample = dataset[0]
        print(f"✓ Sample 0 loaded successfully")
        
        # 检查各字段
        assert sample['images'].shape == (2, 3, 392, 518), \
            f"Image shape mismatch: {sample['images'].shape}"
        print(f"  - Image shape: {sample['images'].shape}")
        
        assert sample['disparity'].shape == (392, 518), \
            f"Disparity shape mismatch: {sample['disparity'].shape}"
        print(f"  - Disparity shape: {sample['disparity'].shape}")
        
        assert isinstance(sample['gt_scale'].item(), float), \
            f"GT scale is not float: {type(sample['gt_scale'])}"
        print(f"  - GT scale: {sample['gt_scale'].item():.6f}")
        
        assert isinstance(sample['calibration'], dict), \
            f"Calibration is not dict: {type(sample['calibration'])}"
        print(f"  - Calibration keys: {list(sample['calibration'].keys())}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        return False


def test_calibration_scaling():
    """测试calibration缩放的数学一致性"""
    print("=" * 60)
    print("TEST 3: Calibration Scaling Verification")
    print("=" * 60)
    
    dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI", indices=[0])
    sample = dataset[0]
    calib = sample['calibration']
    
    K_orig = calib['K_original']
    K_scaled = calib['K_scaled']
    scale_x = calib['scale_x']
    scale_y = calib['scale_y']
    
    print(f"Original K matrix:")
    print(f"  fx={K_orig[0, 0]:.2f}, fy={K_orig[1, 1]:.2f}")
    print(f"  cx={K_orig[0, 2]:.2f}, cy={K_orig[1, 2]:.2f}")
    print()
    
    print(f"Scaled K matrix:")
    print(f"  fx={K_scaled[0, 0]:.2f}, fy={K_scaled[1, 1]:.2f}")
    print(f"  cx={K_scaled[0, 2]:.2f}, cy={K_scaled[1, 2]:.2f}")
    print()
    
    print(f"Scale factors:")
    print(f"  scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
    print()
    
    # 验证焦距缩放
    tolerance = 1e-5
    
    actual_scale_x_fx = K_scaled[0, 0] / K_orig[0, 0]
    actual_scale_y_fy = K_scaled[1, 1] / K_orig[1, 1]
    actual_scale_x_cx = K_scaled[0, 2] / K_orig[0, 2]
    actual_scale_y_cy = K_scaled[1, 2] / K_orig[1, 2]
    
    fx_ok = abs(actual_scale_x_fx - scale_x) < tolerance
    fy_ok = abs(actual_scale_y_fy - scale_y) < tolerance
    cx_ok = abs(actual_scale_x_cx - scale_x) < tolerance
    cy_ok = abs(actual_scale_y_cy - scale_y) < tolerance
    
    print(f"Focal length scaling verification:")
    print(f"  fx scale: {actual_scale_x_fx:.6f} (expected {scale_x:.6f}) {'✓' if fx_ok else '✗'}")
    print(f"  fy scale: {actual_scale_y_fy:.6f} (expected {scale_y:.6f}) {'✓' if fy_ok else '✗'}")
    print()
    
    print(f"Principal point scaling verification:")
    print(f"  cx scale: {actual_scale_x_cx:.6f} (expected {scale_x:.6f}) {'✓' if cx_ok else '✗'}")
    print(f"  cy scale: {actual_scale_y_cy:.6f} (expected {scale_y:.6f}) {'✓' if cy_ok else '✗'}")
    print()
    
    all_ok = fx_ok and fy_ok and cx_ok and cy_ok
    if all_ok:
        print(f"✓ All calibration scaling checks passed!")
    else:
        print(f"✗ Some calibration scaling checks failed!")
    
    print()
    return all_ok


def test_disparity_range():
    """测试disparity范围"""
    print("=" * 60)
    print("TEST 4: Disparity Range")
    print("=" * 60)
    
    dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI", indices=list(range(5)))
    
    all_min = np.inf
    all_max = -np.inf
    valid_counts = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        disp = sample['disparity']
        
        valid_mask = disp > 0
        valid_counts.append(valid_mask.sum())
        
        if valid_mask.sum() > 0:
            all_min = min(all_min, disp[valid_mask].min())
            all_max = max(all_max, disp[valid_mask].max())
    
    print(f"Disparity statistics (5 samples):")
    print(f"  Valid pixel count: min={min(valid_counts)}, max={max(valid_counts)}")
    print(f"  Value range: {all_min:.4f} to {all_max:.4f}")
    print(f"  ✓ Disparity loading verified")
    print()
    
    return True


def test_gt_scale_calculation():
    """测试GT scale标签计算的合理性"""
    print("=" * 60)
    print("TEST 5: GT Scale Calculation")
    print("=" * 60)
    
    dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI", indices=list(range(10)))
    
    scales = []
    for i in range(len(dataset)):
        sample = dataset[i]
        gt_scale = sample['gt_scale'].item()
        scales.append(gt_scale)
    
    scales = np.array(scales)
    
    print(f"GT scale statistics (10 samples):")
    print(f"  Mean: {scales.mean():.6f}")
    print(f"  Std:  {scales.std():.6f}")
    print(f"  Min:  {scales.min():.6f}")
    print(f"  Max:  {scales.max():.6f}")
    print()
    
    # 检查合理性：尺度应在合理范围内（通常在0.001-100之间）
    # 对于KITTI，通常在0.01-1.0之间
    reasonable = np.all((scales > 0.001) & (scales < 100))
    
    if reasonable:
        print(f"✓ All GT scales are within reasonable range [0.001, 100]")
    else:
        print(f"✗ Some GT scales are outside reasonable range!")
        invalid_idx = np.where((scales <= 0.001) | (scales >= 100))[0]
        print(f"  Invalid indices: {invalid_idx}")
    
    print()
    return reasonable


def test_visualization():
    """生成可视化结果"""
    print("=" * 60)
    print("TEST 6: Visualization")
    print("=" * 60)
    
    dataset = KITTIStereoDataset("/home/hba/Documents/Dataset/KITTI", indices=[0, 1, 2])
    
    fig, axes = plt.subplots(len(dataset), 2, figsize=(12, 4 * len(dataset)))
    
    if len(dataset) == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # 显示左图
        img_left = sample['images'][0].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_left)
        axes[i, 0].set_title(f"Sample {i}: Left Image (518×392)")
        axes[i, 0].axis('off')
        
        # 显示disparity
        disp = sample['disparity']
        axes[i, 1].imshow(disp, cmap='jet')
        axes[i, 1].set_title(f"Disparity (scale={sample['gt_scale'].item():.4f})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = ROOT_DIR / "tests" / "kitti_preprocessing_check.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()
    
    print()
    return True


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  KITTI Preprocessing Validation Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Single Sample", test_single_sample),
        ("Calibration Scaling", test_calibration_scaling),
        ("Disparity Range", test_disparity_range),
        ("GT Scale Calculation", test_gt_scale_calculation),
        ("Visualization", test_visualization),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # 总结
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    print()
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
