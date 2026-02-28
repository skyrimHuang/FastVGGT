#!/usr/bin/env python3
"""
单元测试4: KITTI数据集加载测试
支持实际数据加载验证（使用cv2.IMREAD_ANYDEPTH读取uint16）
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import cv2
from PIL import Image


def create_mock_kitti_data(temp_dir: Path, num_samples: int = 3) -> Path:
    """
    创建模拟KITTI数据集目录结构和文件
    
    Args:
        temp_dir: 临时目录
        num_samples: 创建的样本数
    
    Returns:
        KITTI数据根目录路径
    """
    kitti_root = temp_dir / "mock_kitti"
    
    # 创建目录结构
    scene_flow_dir = kitti_root / "data_scene_flow" / "training"
    calib_dir = kitti_root / "data_scene_flow_calib" / "training" / "calib_cam_to_cam"
    
    for subdir in [
        scene_flow_dir / "image_2",
        scene_flow_dir / "image_3",
        scene_flow_dir / "disp_occ_0",
        calib_dir
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
    
    # 创建样本数据
    for i in range(num_samples):
        idx = f"{i:06d}"
        
        # 左右图像 (RGB格式，1242x375)
        img_left = np.random.randint(0, 256, (375, 1242, 3), dtype=np.uint8)
        img_right = np.random.randint(0, 256, (375, 1242, 3), dtype=np.uint8)
        
        cv2.imwrite(str(scene_flow_dir / "image_2" / f"{idx}_10.png"), cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(scene_flow_dir / "image_3" / f"{idx}_10.png"), cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
        
        # Disparity (uint16格式，1242x375，范围0-32000)
        disp = np.random.randint(100, 200, (375, 1242), dtype=np.uint16)
        cv2.imwrite(str(scene_flow_dir / "disp_occ_0" / f"{idx}_10.png"), disp)
        
        # Calibration文件
        focal_length = 721.537  # KITTI标准值
        principal_x = 609.559
        principal_y = 172.854
        baseline = 0.54  # 立体基线（米）
        
        calib_content = f"""P_rect_00: {focal_length} 0 {principal_x} 0 0 {focal_length} {principal_y} 0 0 0 1 0
P_rect_01: {focal_length} 0 {principal_x} {-focal_length * baseline} 0 {focal_length} {principal_y} 0 0 0 1 0
R_rect_00: 1 0 0 0 1 0 0 0 1
T_01: {baseline} 0 0
"""
        calib_path = calib_dir / f"{idx}.txt"
        calib_path.write_text(calib_content)
    
    return kitti_root


def test_imports():
    """测试模块导入"""
    print("TEST 1: 测试模块导入...")
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset, kitti_collate_fn
        from eval.dataset_utils.kitti_calib import KITTICalibrationProcessor
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_initialization():
    """测试数据集初始化API"""
    print("\nTEST 2: 测试数据集初始化...")
    
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset
        
        # 验证类能被正确导入和定义
        assert hasattr(KITTIStereoDataset, '__init__'), "缺少__init__方法"
        assert hasattr(KITTIStereoDataset, '__getitem__'), "缺少__getitem__方法"
        assert hasattr(KITTIStereoDataset, '__len__'), "缺少__len__方法"
        
        print(f"  ✓ class structure verified")
        print("✓ 数据集初始化API正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_with_actual_data():
    """测试使用真实模拟数据进行加载"""
    print("\nTEST 3: 测试实际数据加载...")
    
    temp_dir = None
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset
        
        # 创建模拟数据
        temp_dir = Path(tempfile.mkdtemp())
        kitti_root = create_mock_kitti_data(temp_dir, num_samples=3)
        
        # 初始化数据集
        dataset = KITTIStereoDataset(
            data_dir=str(kitti_root),
            indices=[0, 1],  # 加载前2个样本
            enable_calib_cache=True,
            resize_interpolation='cubic'
        )
        
        # 验证数据集长度
        assert len(dataset) == 2, f"数据集长度应为2，实际{len(dataset)}"
        
        # 加载第一个样本
        sample = dataset[0]
        
        # 验证返回的键
        required_keys = ['images', 'disparity', 'calibration', 'gt_scale', 'metadata']
        for key in required_keys:
            assert key in sample, f"缺少键: {key}"
        
        # 验证数据形状和类型
        images = sample['images']
        assert isinstance(images, torch.Tensor), "images应为torch.Tensor"
        assert images.shape[0] == 2, "应有左右两帧图像"
        assert images.shape[1] == 3, "应有RGB三个通道"
        assert images.shape[2:4] == (392, 518), f"图像形状错误: {images.shape[2:4]}"
        
        # 验证disparity
        disp = sample['disparity']
        assert isinstance(disp, np.ndarray), "disparity应为numpy数组"
        assert disp.dtype in [np.float32, np.float64], f"disparity数据类型错误: {disp.dtype}"
        assert disp.shape == (392, 518), f"disparity形状错误: {disp.shape}"
        
        # 验证GT scale
        gt_scale = sample['gt_scale']
        assert isinstance(gt_scale, torch.Tensor), "gt_scale应为torch.Tensor"
        assert gt_scale.item() > 0, "gt_scale应为正数"
        
        # 验证calibration
        calib = sample['calibration']
        assert 'K_original' in calib, "缺少K_original"
        assert 'K_scaled' in calib, "缺少K_scaled"
        assert 'baseline' in calib, "缺少baseline"
        
        print(f"  图像形状: {images.shape}")
        print(f"  视差形状: {disp.shape}")
        print(f"  GT scale: {gt_scale.item():.4f}")
        print(f"  Baseline: {calib['baseline']:.4f}m")
        print("✓ 实际数据加载成功")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir)


def test_collate_fn_with_batch():
    """测试collate_fn与实际batch数据"""
    print("\nTEST 4: 测试collate_fn与batch...")
    
    temp_dir = None
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset, kitti_collate_fn
        from torch.utils.data import DataLoader
        
        # 创建模拟数据
        temp_dir = Path(tempfile.mkdtemp())
        kitti_root = create_mock_kitti_data(temp_dir, num_samples=4)
        
        # 初始化数据集和DataLoader
        dataset = KITTIStereoDataset(
            data_dir=str(kitti_root),
            indices=[0, 1, 2],
            enable_calib_cache=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=kitti_collate_fn
        )
        
        # 获取第一个batch
        batch = next(iter(loader))
        
        # 验证batch结构
        assert 'images' in batch, "batch缺少images"
        assert batch['images'].shape[0] == 2, "batch_size应为2"
        assert batch['images'].shape[1] == 2, "应有左右两帧"
        
        assert 'gt_scale' in batch, "batch缺少gt_scale"
        assert batch['gt_scale'].shape[0] == 2, "gt_scale batch_size应为2"
        
        print(f"  Batch图像形状: {batch['images'].shape}")
        print(f"  Batch GT scale形状: {batch['gt_scale'].shape}")
        print("✓ Collate函数工作正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir)


def test_dataloader_integration():
    """测试DataLoader API兼容性"""
    print("\nTEST 5: 测试DataLoader兼容性...")
    
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset, kitti_collate_fn
        from torch.utils.data import DataLoader
        import inspect
        
        # 验证KITTIStereoDataset与DataLoader兼容（实现了必要的接口）
        assert hasattr(KITTIStereoDataset, '__len__'), "缺少__len__方法"
        assert hasattr(KITTIStereoDataset, '__getitem__'), "缺少__getitem__方法"
        
        # 验证collate_fn签名
        sig = inspect.signature(kitti_collate_fn)
        assert len(sig.parameters) > 0, "collate_fn应该接受batch参数"
        
        print(f"  ✓ KITTIStereoDataset implements __len__ and __getitem__")
        print(f"  ✓ kitti_collate_fn is callable")
        print("✓ DataLoader兼容性验证通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_disparity_handling():
    """测试无效视差的处理"""
    print("\nTEST 6: 测试无效视差处理...")
    
    temp_dir = None
    try:
        from eval.dataset_utils.kitti_stereo import KITTIStereoDataset
        
        # 创建模拟数据（包含零视差）
        temp_dir = Path(tempfile.mkdtemp())
        kitti_root = create_mock_kitti_data(temp_dir, num_samples=2)
        
        # 修改第一个样本的disparity，包含无效值（0表示无效）
        disp_path = kitti_root / "data_scene_flow" / "training" / "disp_occ_0" / "000000_10.png"
        disp = np.zeros((375, 1242), dtype=np.uint16)
        disp[100:200, 100:200] = 150  # 只有一部分有效
        cv2.imwrite(str(disp_path), disp)
        
        # 加载数据集
        dataset = KITTIStereoDataset(
            data_dir=str(kitti_root),
            indices=[0],
            enable_calib_cache=True
        )
        
        # 获取样本
        sample = dataset[0]
        
        # 验证gt_scale仍然有效
        gt_scale = sample['gt_scale']
        assert gt_scale.item() >= 0, "gt_scale不应为负"
        assert not torch.isnan(gt_scale), "gt_scale不应为NaN"
        
        print(f"  GT scale (with invalid disparity): {gt_scale.item():.4f}")
        print("✓ 无效视差处理成功")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir)


def main():
    """主测试函数"""
    print("=" * 60)
    print("单元测试4: KITTI数据集加载（支持实际数据加载）")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_dataset_initialization,
        test_dataset_with_actual_data,
        test_collate_fn_with_batch,
        test_dataloader_integration,
        test_invalid_disparity_handling,
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
