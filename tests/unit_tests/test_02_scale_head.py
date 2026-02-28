#!/usr/bin/env python3
"""
单元测试2: Scale Head模块测试
测试尺度预测头的结构和前向传播
"""

import sys
from pathlib import Path

# 添加项目根目录（tests/unit_tests/test_02... -> tests/unit_tests/ -> tests/ -> FastVGGT/）
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn


def test_imports():
    """测试模块导入"""
    print("TEST 1: 测试模块导入...")
    try:
        from vggt.heads.scale_head import (
            KITTIStereoScaleHead,
            SimpleStereoScaleHead,
            create_scale_head
        )
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kitti_scale_head_structure():
    """测试KITTIStereoScaleHead结构"""
    print("\nTEST 2: 测试KITTIStereoScaleHead结构...")
    try:
        from vggt.heads.scale_head import KITTIStereoScaleHead
        
        # 创建模型
        model = KITTIStereoScaleHead(
            dim_in=2048,
            hidden_dims=[1024, 512],
            dropout=0.1,
            use_calibration_features=True
        )
        
        # 检查属性
        assert model.dim_in == 2048
        assert model.hidden_dims == [1024, 512]
        assert model.use_calibration_features == True
        
        # 检查模型有mlp
        assert hasattr(model, 'mlp')
        assert isinstance(model.mlp, nn.Sequential)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数量: {total_params:,}")
        print(f"  输入维度: {model.dim_in}")
        print(f"  隐层维度: {model.hidden_dims}")
        print("✓ 结构正确")
        return True
    except Exception as e:
        print(f"✗ 结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kitti_scale_head_forward():
    """测试KITTIStereoScaleHead前向传播"""
    print("\nTEST 3: 测试KITTIStereoScaleHead前向传播...")
    try:
        from vggt.heads.scale_head import KITTIStereoScaleHead
        
        model = KITTIStereoScaleHead(
            dim_in=2048,
            hidden_dims=[1024, 512],
            use_calibration_features=True
        )
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        num_frames = 2  # 立体对
        num_patches = 37 * 28  # 518/14 * 392/14
        dim = 2048
        
        tokens = torch.randn(batch_size, num_frames, num_patches, dim)
        calib_features = torch.tensor([
            [0.54, 300.0],  # baseline=0.54m, focal=300
            [0.54, 300.0]
        ])
        
        # 前向传播
        with torch.no_grad():
            scale_output = model(tokens, calib_features)
        
        # 验证输出
        assert scale_output.shape == (batch_size, 1), f"输出形状错误: {scale_output.shape}"
        assert torch.all(scale_output > 0), "scale必须为正数"
        assert not torch.any(torch.isnan(scale_output)), "输出包含NaN"
        assert not torch.any(torch.isinf(scale_output)), "输出包含Inf"
        
        print(f"  输入形状: {tuple(tokens.shape)}")
        print(f"  输出形状: {tuple(scale_output.shape)}")
        print(f"  输出值: {scale_output.squeeze().tolist()}")
        print("✓ 前向传播正确")
        return True
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_scale_head():
    """测试SimpleStereoScaleHead"""
    print("\nTEST 4: 测试SimpleStereoScaleHead...")
    try:
        from vggt.heads.scale_head import SimpleStereoScaleHead
        
        model = SimpleStereoScaleHead(
            dim_in=2048,
            hidden_dims=[1024, 512]
        )
        model.eval()
        
        # 创建测试输入（不需要calibration）
        tokens = torch.randn(2, 2, 1036, 2048)
        
        with torch.no_grad():
            scale_output = model(tokens)
        
        assert scale_output.shape == (2, 1)
        assert torch.all(scale_output > 0)
        
        print(f"  输出形状: {tuple(scale_output.shape)}")
        print(f"  输出值: {scale_output.squeeze().tolist()}")
        print("✓ SimpleStereoScaleHead正确")
        return True
    except Exception as e:
        print(f"✗ SimpleStereoScaleHead测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_scale_head_factory():
    """测试工厂函数"""
    print("\nTEST 5: 测试create_scale_head工厂函数...")
    try:
        from vggt.heads.scale_head import (
            create_scale_head,
            KITTIStereoScaleHead,
            SimpleStereoScaleHead
        )
        
        # 测试创建KITTI版本
        model_kitti = create_scale_head('kitti', dim_in=2048)
        assert isinstance(model_kitti, KITTIStereoScaleHead)
        
        # 测试创建Simple版本
        model_simple = create_scale_head('simple', dim_in=2048)
        assert isinstance(model_simple, SimpleStereoScaleHead)
        
        # 测试错误类型
        try:
            model_invalid = create_scale_head('invalid_type', dim_in=2048)
            assert False, "应该抛出ValueError"
        except ValueError:
            pass
        
        print("✓ 工厂函数正确")
        return True
    except Exception as e:
        print(f"✗ 工厂函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """测试梯度流"""
    print("\nTEST 6: 测试梯度流...")
    try:
        from vggt.heads.scale_head import KITTIStereoScaleHead
        
        model = KITTIStereoScaleHead(dim_in=2048)
        model.train()
        
        # 创建输入
        tokens = torch.randn(1, 2, 100, 2048, requires_grad=True)
        calib_features = torch.tensor([[0.54, 300.0]])
        target = torch.tensor([[10.0]])
        
        # 前向传播
        output = model(tokens, calib_features)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(torch.log(output), torch.log(target))
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        assert tokens.grad is not None, "输入没有梯度"
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数{name}没有梯度"
                assert not torch.any(torch.isnan(param.grad)), f"参数{name}梯度包含NaN"
        
        print(f"  损失值: {loss.item():.6f}")
        print(f"  输入梯度范数: {tokens.grad.norm().item():.6f}")
        print("✓ 梯度流正常")
        return True
    except Exception as e:
        print(f"✗ 梯度流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("单元测试: Scale Head模块")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_kitti_scale_head_structure,
        test_kitti_scale_head_forward,
        test_simple_scale_head,
        test_create_scale_head_factory,
        test_gradient_flow,
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
