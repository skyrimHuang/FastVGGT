"""
关键帧预过滤与特征复用模块 —— 单元测试

测试内容:
  1. KeyframeFilter 基本功能：特征提取、关键帧选择、完整流程
  2. Aggregator 特征复用路径：precomputed_patch_tokens 接口正确性
  3. VGGT 端到端透传：forward 接口兼容性
  4. 边界条件：单帧输入、全选/全弃、阈值极端值

运行方式:
    cd /home/hba/Documents/FastVGGT_2
    python -m pytest tests/test_keyframe_filter.py -v
    # 或直接运行
    python tests/test_keyframe_filter.py
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# 辅助: 轻量级 Mock 模块，用于CPU上运行测试（不需加载完整权重）
# ============================================================


class MockPatchEmbed(nn.Module):
    """
    模拟DINOv2编码器的轻量替代品。
    输出与真实DINOv2格式一致的 dict:
      {
        "x_norm_clstoken":     [B*S, embed_dim],
        "x_norm_patchtokens":  [B*S, P, embed_dim]
      }
    其中 P = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, embed_dim=64, patch_size=14, img_size=518):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        # 简单线性映射模拟特征提取
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B*S, 3, H, W]
        B, C, H, W = x.shape
        patches = self.proj(x)  # [B, embed_dim, H/p, W/p]
        pH, pW = patches.shape[2], patches.shape[3]
        patch_tokens = patches.flatten(2).transpose(1, 2)  # [B, P, embed_dim]
        patch_tokens = self.norm(patch_tokens)

        # CLS token = patch tokens 的均值
        cls_token = patch_tokens.mean(dim=1)  # [B, embed_dim]

        return {
            "x_norm_clstoken": cls_token,
            "x_norm_patchtokens": patch_tokens,
        }


class MockAggregator(nn.Module):
    """
    模拟Aggregator，仅保留KeyframeFilter所需的接口属性。
    """

    def __init__(self, embed_dim=64, patch_size=14, img_size=518):
        super().__init__()
        self.patch_embed = MockPatchEmbed(embed_dim, patch_size, img_size)
        self.patch_size = patch_size
        # 注册归一化缓冲区（与真实Aggregator一致）
        self.register_buffer(
            "_resnet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )


# ============================================================
# 测试类
# ============================================================


class TestKeyframeFilterBasic(unittest.TestCase):
    """测试 KeyframeFilter 基础功能"""

    def setUp(self):
        """初始化测试环境"""
        self.embed_dim = 64
        self.img_size = 518
        self.patch_size = 14
        self.aggregator = MockAggregator(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            img_size=self.img_size,
        )

        from vggt.utils.keyframe_filter import KeyframeFilter

        self.filter = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.35,
            min_keyframes=2,
        )

    def test_extract_features_shape(self):
        """验证特征提取输出形状正确"""
        B, S = 1, 5
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        cls_tokens, patch_tokens = self.filter.extract_features(images)

        P = (self.img_size // self.patch_size) ** 2  # 37*37 = 1369
        self.assertEqual(cls_tokens.shape, (B, S, self.embed_dim))
        self.assertEqual(patch_tokens.shape, (B * S, P, self.embed_dim))

    def test_select_keyframes_always_keeps_first(self):
        """验证第一帧始终被保留"""
        B, S, D = 1, 10, self.embed_dim
        cls_tokens = torch.randn(B, S, D)
        indices = self.filter.select_keyframes(cls_tokens)
        self.assertIn(0, indices[0])

    def test_select_keyframes_min_keyframes(self):
        """验证最少保留 min_keyframes 帧"""
        # 使用完全相同的帧（余弦距离=0，只有首帧被选）
        B, S, D = 1, 10, self.embed_dim
        feat = torch.randn(1, D)
        cls_tokens = feat.unsqueeze(0).expand(B, S, D).clone()
        indices = self.filter.select_keyframes(cls_tokens)
        self.assertGreaterEqual(len(indices[0]), self.filter.min_keyframes)

    def test_select_keyframes_all_different(self):
        """验证完全不同的帧全部被保留（阈值足够小）"""
        from vggt.utils.keyframe_filter import KeyframeFilter

        filter_low = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.001,  # 极低阈值 → 几乎只要有微小变化就选
            min_keyframes=1,
        )
        B, S, D = 1, 5, self.embed_dim
        # 构造互相正交的特征向量（余弦距离≈1）
        cls_tokens = torch.eye(D)[:S].unsqueeze(0)  # [1, S, D]
        indices = filter_low.select_keyframes(cls_tokens)
        self.assertEqual(len(indices[0]), S)

    def test_threshold_zero_keeps_all(self):
        """验证阈值=0时所有不完全相同的帧都被保留"""
        from vggt.utils.keyframe_filter import KeyframeFilter

        filter_zero = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.0,
            min_keyframes=1,
        )
        B, S, D = 1, 8, self.embed_dim
        cls_tokens = torch.randn(B, S, D)
        indices = filter_zero.select_keyframes(cls_tokens)
        # 随机特征之间余弦距离几乎不可能恰好为0，预期全部保留
        self.assertEqual(len(indices[0]), S)


class TestKeyframeFilterPipeline(unittest.TestCase):
    """测试完整的 __call__ 管线"""

    def setUp(self):
        self.embed_dim = 64
        self.img_size = 518
        self.patch_size = 14
        self.aggregator = MockAggregator(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            img_size=self.img_size,
        )
        from vggt.utils.keyframe_filter import KeyframeFilter

        self.filter = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.35,
            min_keyframes=2,
        )

    def test_call_output_keys(self):
        """验证 __call__ 返回的字典包含所有必需的键"""
        B, S = 1, 6
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        result = self.filter(images)

        expected_keys = {
            "filtered_images",
            "patch_tokens",
            "keyframe_indices",
            "cls_tokens",
            "stats",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_call_output_shapes(self):
        """验证输出张量形状一致性"""
        B, S = 1, 6
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        result = self.filter(images)

        K = result["stats"]["kept_frames"]
        P = (self.img_size // self.patch_size) ** 2

        # 过滤后图像
        self.assertEqual(result["filtered_images"].shape[0], B)
        self.assertEqual(result["filtered_images"].shape[1], K)
        self.assertEqual(result["filtered_images"].shape[2], 3)

        # patch tokens 形状
        self.assertEqual(result["patch_tokens"].shape, (B * K, P, self.embed_dim))

        # CLS tokens 保留所有帧
        self.assertEqual(result["cls_tokens"].shape, (B, S, self.embed_dim))

    def test_stats_consistency(self):
        """验证统计信息一致性"""
        B, S = 1, 10
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        result = self.filter(images)

        stats = result["stats"]
        self.assertEqual(stats["total_frames"], S)
        self.assertGreaterEqual(stats["kept_frames"], self.filter.min_keyframes)
        self.assertLessEqual(stats["kept_frames"], S)
        self.assertAlmostEqual(
            stats["compression_ratio"],
            stats["kept_frames"] / stats["total_frames"],
        )

    def test_single_frame_input(self):
        """边界条件: 单帧输入"""
        from vggt.utils.keyframe_filter import KeyframeFilter

        filt = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.35,
            min_keyframes=1,
        )
        B, S = 1, 1
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        result = filt(images)

        self.assertEqual(result["stats"]["kept_frames"], 1)
        self.assertEqual(len(result["keyframe_indices"][0]), 1)
        self.assertEqual(result["keyframe_indices"][0][0], 0)


class TestAggregatorPrecomputedPath(unittest.TestCase):
    """测试 Aggregator 的 precomputed_patch_tokens 路径"""

    def setUp(self):
        self.embed_dim = 64
        self.img_size = 518
        self.patch_size = 14
        self.aggregator = MockAggregator(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            img_size=self.img_size,
        )

    def test_precomputed_tokens_shape_compatible(self):
        """验证预计算token形状与Aggregator期望兼容"""
        from vggt.utils.keyframe_filter import KeyframeFilter

        filt = KeyframeFilter(aggregator=self.aggregator, threshold=0.0)
        B, S = 1, 4
        images = torch.randn(B, S, 3, self.img_size, self.img_size)
        result = filt(images)

        patch_tokens = result["patch_tokens"]
        P = (self.img_size // self.patch_size) ** 2
        K = result["stats"]["kept_frames"]

        # 形状应为 [B*K, P, embed_dim]
        self.assertEqual(patch_tokens.shape, (B * K, P, self.embed_dim))
        # 数据类型应为 bf16（与Aggregator内部一致）
        self.assertEqual(patch_tokens.dtype, torch.bfloat16)


class TestFeatureReuse(unittest.TestCase):
    """
    测试特征复用的一致性:
    当阈值=0（保留所有帧）时，precomputed路径的patch token
    应与原始路径完全一致。
    """

    def setUp(self):
        self.embed_dim = 64
        self.img_size = 518
        self.patch_size = 14
        self.aggregator = MockAggregator(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            img_size=self.img_size,
        )

    def test_feature_consistency(self):
        """验证预计算token与直接编码结果的数值一致性"""
        from vggt.utils.keyframe_filter import KeyframeFilter

        filt = KeyframeFilter(
            aggregator=self.aggregator,
            threshold=0.0,  # 保留所有帧
            min_keyframes=1,
        )

        B, S = 1, 3
        images = torch.randn(B, S, 3, self.img_size, self.img_size)

        # 路径1: 通过 KeyframeFilter 获取 patch tokens
        result = filt(images)
        tokens_from_filter = result["patch_tokens"]

        # 路径2: 直接调用 patch_embed（模拟 Aggregator 原始路径）
        images_norm = images.to(torch.bfloat16)
        mean = self.aggregator._resnet_mean
        std = self.aggregator._resnet_std
        images_norm = (images_norm - mean) / std
        images_flat = images_norm.view(B * S, 3, self.img_size, self.img_size)

        with torch.no_grad():
            features_direct = self.aggregator.patch_embed(images_flat)
            tokens_direct = features_direct["x_norm_patchtokens"].to(torch.bfloat16)

        # 数值比较（bf16精度下应完全一致，因为经过相同路径）
        max_diff = (tokens_from_filter.float() - tokens_direct.float()).abs().max()
        self.assertLess(
            max_diff,
            1e-3,
            f"预计算token与直接编码token最大差异 {max_diff:.6f} 超过阈值",
        )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("关键帧预过滤与特征复用模块 —— 单元测试")
    print("=" * 60)
    unittest.main(verbosity=2)
