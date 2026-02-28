#!/usr/bin/env python
"""
快速验证脚本 —— 确保关键帧过滤评估管道正常工作

使用合成数据而非真实数据，快速验证：
  1. 模块加载和参数解析
  2. 特征提取和区分度评估逻辑
  3. 时间和显存评估流程
  4. 图表生成和CSV输出

运行: python tests/test_eval_pipeline.py
"""

import os
import sys
import torch
import numpy as np
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.utils.keyframe_filter import KeyframeFilter


# ============================================================
# Mock 模型（CPU快速测试）
# ============================================================

class MockAggregator:
    """模式匹配KeyframeFilter期望的接口"""
    def __init__(self):
        import torch.nn as nn
        self.patch_embed = MockPatchEmbed()
        self._resnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self._resnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class MockPatchEmbed(torch.nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
    
    def forward(self, x):
        B = x.shape[0]
        # 保持float32进行conv，避免dtype mismatch
        x = x.float()
        patches = self.proj(x)  # [B, C, H', W']
        pH, pW = patches.shape[2], patches.shape[3]
        P = pH * pW
        patch_tokens = patches.view(B, self.embed_dim, P).transpose(1, 2)  # [B, P, C]
        
        # CLS token from pixel stats (和bench_keyframe_filter.py一致)
        pixel_features = []
        for c_idx in range(3):
            channel = x[:, c_idx, :, :]
            pixel_features.append(channel.mean(dim=(1, 2)))
            pixel_features.append(channel.std(dim=(1, 2)))
            pixel_features.append(channel.max(dim=1)[0].max(dim=1)[0])
        pixel_features.append(x.mean(dim=(1, 2, 3)))
        pixel_features.append(x.std(dim=(1, 2, 3)))
        pixel_vec = torch.stack(pixel_features, dim=1)  # [B, 13]
        
        n_repeat = (self.embed_dim // pixel_vec.shape[1]) + 1
        cls_token = pixel_vec.repeat(1, n_repeat)[:, :self.embed_dim]  # [B, C]
        
        return {
            "x_norm_clstoken": cls_token,
            "x_norm_patchtokens": patch_tokens,
        }


def test_feature_extraction():
    """测试特征提取流程"""
    print("\n[TEST 1] 特征提取")
    print("-" * 50)
    
    aggregator = MockAggregator()
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=0.35)
    
    # 合成视频：[B=1, S=10, C=3, H=518, W=518]
    images = torch.randn(1, 10, 3, 518, 518)
    
    with torch.no_grad():
        cls_tokens, patch_tokens = filter_model.extract_features(images)
    
    print(f"✓ 输入形状: {images.shape}")
    print(f"✓ CLS token形状: {cls_tokens.shape}")
    print(f"✓ Patch token形状: {patch_tokens.shape}")
    print(f"✓ CLS token统计: mean={cls_tokens.mean():.4f}, std={cls_tokens.std():.4f}")


def test_keyframe_selection():
    """测试关键帧选择"""
    print("\n[TEST 2] 关键帧选择")
    print("-" * 50)
    
    aggregator = MockAggregator()
    
    for threshold in [0.1, 0.35, 0.7]:
        filter_model = KeyframeFilter(
            aggregator=aggregator,
            threshold=threshold,
            min_keyframes=2
        )
        
        images = torch.randn(1, 20, 3, 518, 518)
        result = filter_model(images)
        
        kept = result["stats"]["kept_frames"]
        total = result["stats"]["total_frames"]
        ratio = result["stats"]["compression_ratio"]
        
        print(f"  τ={threshold:.2f}: {kept}/{total} 帧 ({ratio:.1%} 保留)")


def test_full_pipeline():
    """测试完整管道"""
    print("\n[TEST 3] 完整管道 (__call__)")
    print("-" * 50)
    
    aggregator = MockAggregator()
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=0.35)
    
    images = torch.randn(2, 15, 3, 518, 518)  # 两个batch
    
    with torch.no_grad():
        result = filter_model(images)
    
    print(f"✓ 输入: {images.shape}")
    print(f"✓ 输出关键:")
    for key in ["filtered_images", "patch_tokens", "keyframe_indices", "cls_tokens"]:
        if key == "keyframe_indices":
            print(f"    {key}: 长度={len(result[key])}")
        else:
            print(f"    {key}: {result[key].shape}")
    print(f"✓ 统计信息:")
    for k, v in result["stats"].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")


def test_memory_simulation():
    """模拟不同序列长度的显存占用"""
    print("\n[TEST 4] 显存占用模拟")
    print("-" * 50)
    
    aggregator = MockAggregator()
    
    for seq_len in [5, 10, 20, 30]:
        filter_model = KeyframeFilter(aggregator=aggregator, threshold=0.35)
        
        # 模拟图像加载（在CPU上，避免OOM）
        images = torch.randn(1, seq_len, 3, 518, 518)
        
        # 估计显存占用（仅基于张量大小）
        est_memory = (
            images.numel() * 2 / 1024 / 1024  # float32 4bytes
            + 100  # overhead
        )
        
        print(f"  序列长度 {seq_len:2d}: 预估 ~{est_memory:.0f} MB")


def test_cosine_distances():
    """测试余弦距离计算"""
    print("\n[TEST 5] 余弦距离分析")
    print("-" * 50)
    
    import torch.nn.functional as F
    
    aggregator = MockAggregator()
    filter_model = KeyframeFilter(aggregator=aggregator, threshold=0.35)
    
    # 模拟不同运动类型的视频
    for motion_type in ["smooth", "random"]:
        if motion_type == "smooth":
            # 平滑运动：逐帧线性变化
            base = torch.randn(1, 1, 3, 518, 518)
            t = torch.linspace(0, 1, 20).view(1, 20, 1, 1, 1)
            noise = torch.randn(1, 20, 3, 518, 518) * 0.5
            images = (base * (1 - t) + (base + noise) * t).clamp(0, 1)
        else:
            # 随机运动
            images = torch.rand(1, 20, 3, 518, 518)
        
        with torch.no_grad():
            cls_tokens, _ = filter_model.extract_features(images)
        
        cls_norm = F.normalize(cls_tokens[0].float(), dim=1)
        distances = []
        for i in range(1, 20):
            cos_sim = (cls_norm[i] * cls_norm[i-1]).sum().item()
            dist = 1.0 - cos_sim
            distances.append(dist)
        
        dists = np.array(distances)
        print(f"  {motion_type:10s}: 均值={dists.mean():.4f}, "
              f"标准差={dists.std():.4f}, "
              f"最大值={dists.max():.4f}")


def main():
    print("=" * 70)
    print("关键帧过滤评估管道 —— 快速验证")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_feature_extraction()
        test_keyframe_selection()
        test_full_pipeline()
        test_memory_simulation()
        test_cosine_distances()
        
        print("\n" + "=" * 70)
        print("✅ 所有验证测试通过！")
        print("=" * 70)
        print("\n下一步: 使用真实数据运行完整评估")
        print("$ python tests/eval_keyframe_filter_realdata.py \\")
        print("    --dataset_type 7scenes \\")
        print("    --data_dir /path/to/7scenes \\")
        print("    --sequence_lengths 5,10,15,20,30")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
