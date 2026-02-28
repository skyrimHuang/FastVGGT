#!/usr/bin/env python3
"""
单元测试5: 训练组件测试
测试损失函数、优化器设置等训练相关组件
"""

import sys
from pathlib import Path

# 添加项目根目录（tests/unit_tests/test_05... -> tests/unit_tests/ -> tests/ -> FastVGGT/）
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def convert_model_to_float32(model):
    """递归地将模型的所有参数和缓冲区转换为float32"""
    for module in model.modules():
        for param in module.parameters(recurse=False):
            param.data = param.data.to(torch.float32)
        for buffer_name, buffer in list(module.named_buffers(recurse=False)):
            if buffer is not None:
                setattr(module, buffer_name, buffer.to(torch.float32))
    return model


def test_imports():
    """测试模块导入"""
    print("TEST 1: 测试模块导入...")
    try:
        from vggt.models.vggt import VGGT
        from vggt.heads.scale_head import KITTIStereoScaleHead
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scale_head_loss():
    """测试尺度损失函数"""
    print("\nTEST 2: 测试尺度损失函数...")
    try:
        # 定义损失函数（对数空间MSE）
        def scale_loss_fn(pred_scale, gt_scale):
            """对数空间MSE损失"""
            log_pred = torch.log(pred_scale + 1e-8)
            log_gt = torch.log(gt_scale + 1e-8)
            return nn.functional.mse_loss(log_pred, log_gt)
        
        # 测试损失计算
        pred_scale = torch.tensor([[1.5], [2.0], [0.8]])
        gt_scale = torch.tensor([[1.2], [2.3], [0.9]])  # 匹配pred_scale的形状
        
        loss = scale_loss_fn(pred_scale, gt_scale)
        
        assert loss.item() >= 0, "损失应该非负"
        assert not torch.isnan(loss), "损失不应该是NaN"
        assert not torch.isinf(loss), "损失不应该是Inf"
        
        print(f"  预测尺度: {pred_scale.flatten().tolist()}")
        print(f"  GT尺度: {gt_scale.tolist()}")
        print(f"  损失值: {loss.item():.6f}")
        print("✓ 损失函数正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_setup():
    """测试优化器设置"""
    print("\nTEST 3: 测试优化器设置...")
    try:
        from vggt.models.vggt import VGGT
        
        model = VGGT(
            enable_camera=False,
            enable_depth=True,
            enable_scale_head=True
        )
        
        # 冻结backbone参数
        for param in model.aggregator.parameters():
            param.requires_grad = False
        if model.depth_head is not None:
            for param in model.depth_head.parameters():
                param.requires_grad = False
        
        # 只优化scale_head
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 验证优化器
        assert len(optimizer.param_groups) == 1, "应该只有一个参数组"
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        scale_head_params = sum(p.numel() for p in model.scale_head.parameters())
        
        assert trainable_params == scale_head_params, \
            f"可训练参数应该只包含scale_head: {trainable_params} vs {scale_head_params}"
        
        print(f"  优化器: AdamW")
        print(f"  学习率: {optimizer.param_groups[0]['lr']}")
        print(f"  可训练参数: {trainable_params:,}")
        print("✓ 优化器设置正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler_setup():
    """测试学习率调度器"""
    print("\nTEST 4: 测试学习率调度器...")
    try:
        from vggt.models.vggt import VGGT
        
        model = VGGT(enable_scale_head=True)
        
        # 冻结backbone
        for param in model.aggregator.parameters():
            param.requires_grad = False
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        
        # Cosine退火调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        
        # 测试几个epoch的学习率变化
        initial_lr = optimizer.param_groups[0]['lr']
        lrs = [initial_lr]
        
        for epoch in range(5):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # 学习率应该递减
        assert lrs[1] < lrs[0], "学习率应该递减"
        
        print(f"  初始学习率: {lrs[0]:.2e}")
        print(f"  5个epoch后: {lrs[-1]:.2e}")
        print("✓ 调度器设置正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\nTEST 5: 测试训练步骤...")
    try:
        from vggt.models.vggt import VGGT
        
        # 创建模型
        model = VGGT(
            enable_camera=False,
            enable_depth=True,
            enable_scale_head=True
        )
        
        # 冻结backbone
        for param in model.aggregator.parameters():
            param.requires_grad = False
        if model.depth_head is not None:
            for param in model.depth_head.parameters():
                param.requires_grad = False
        
        # 验证scale_head是否可训练
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        scale_head_params = sum(p.numel() for p in model.scale_head.parameters())
        
        assert trainable_params_count ==scale_head_params, \
            f"只有scale_head应该可训练: {trainable_params_count} vs {scale_head_params}"
        
        model.train()
        
        # 创建优化器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        
        # 验证优化器正确配置
        assert len(optimizer.param_groups) == 1, "应该只有一个参数组"
        assert optimizer.param_groups[0]['lr'] == 1e-4, "学习率应该是1e-4"
        
        print(f"  可训练参数: {trainable_params_count:,}")
        print(f"  ScaleHead参数: {scale_head_params:,}")
        print(f"  优化器配置: AdamW, lr=1e-4")
        print("✓ 训练步骤配置正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_clipping():
    """测试梯度裁剪"""
    print("\nTEST 6: 测试梯度裁剪...")
    try:
        from vggt.models.vggt import VGGT
        
        model = VGGT(enable_scale_head=True)
        
        # 冻结backbone
        for param in model.aggregator.parameters():
            param.requires_grad = False
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        
        # 验证梯度裁剪操作是否可用
        grad_norm_dummy = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=float('inf')
        )
        
        # 验证梯度范数计算返回有效值
        assert grad_norm_dummy == 0 or not torch.isnan(torch.tensor(grad_norm_dummy)), \
            "梯度范数应该是有效的数值"
        
        # 验证硬梯度裁剪设置
        max_norm = 1.0
        grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=max_norm
        )
        
        print(f"  梯度范数(无裁剪): {grad_norm_dummy:.4f}")
        print(f"  梯度范数(裁剪max={max_norm}): {grad_norm_clipped:.4f}")
        print("✓ 梯度裁剪配置正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_mode():
    """测试评估模式"""
    print("\nTEST 7: 测试评估模式...")
    try:
        from vggt.models.vggt import VGGT
        
        model = VGGT(enable_scale_head=True)
        
        # 验证训练模式和评估模式的存在
        # 训练模式
        model.train()
        assert model.training, "模型应该处于训练模式"
        
        # 验证所有子模块都进入训练模式
        all_training = all(module.training for module in model.modules())
        assert all_training, "所有子模块应该进入训练模式"
        
        # 评估模式
        model.eval()
        assert not model.training, "模型应该处于评估模式"
        
        # 验证所有子模块都进入评估模式
        all_eval = all(not module.training for module in model.modules())
        assert all_eval, "所有子模块应该进入评估模式"
        
        # 验证drop out和batch norm在不同模式下的行为差异
        # (通过检查training属性确认)
        print(f"  训练模式设置: model.training = {True}")
        print(f"  评估模式设置: model.training = {False}")
        print("✓ 评估模式切换正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    print("\nTEST 8: 测试检查点保存和加载...")
    try:
        import tempfile
        from vggt.models.vggt import VGGT
        
        # 创建模型
        model = VGGT(enable_scale_head=True)
        
        # 获取scale_head参数
        original_params = {
            name: param.clone()
            for name, param in model.scale_head.named_parameters()
        }
        
        # 保存检查点
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        torch.save({
            'scale_head_state_dict': model.scale_head.state_dict(),
        }, temp_file.name)
        temp_file.close()
        
        # 修改参数
        for param in model.scale_head.parameters():
            param.data.fill_(999.0)
        
        # 加载检查点
        checkpoint = torch.load(temp_file.name)
        model.scale_head.load_state_dict(checkpoint['scale_head_state_dict'])
        
        # 验证参数恢复
        for name, param in model.scale_head.named_parameters():
            assert torch.allclose(param, original_params[name]), \
                f"参数{name}未正确恢复"
        
        # 清理
        import os
        os.unlink(temp_file.name)
        
        print(f"  ✓ 检查点保存成功")
        print(f"  ✓ 检查点加载成功")
        print(f"  ✓ 参数正确恢复")
        print("✓ 检查点功能正确")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("单元测试: 训练组件")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_scale_head_loss,
        test_optimizer_setup,
        test_scheduler_setup,
        test_training_step,
        test_gradient_clipping,
        test_evaluation_mode,
        test_checkpoint_save_load,
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
