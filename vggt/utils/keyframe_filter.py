"""
基于DINOv2空间特征的关键帧选择与特征复用模块。

在视频流进入VGGT网络之前，利用VGGT内部已有的DINOv2编码器提取CLS Token，
通过余弦相似度距离判别关键帧，并缓存被保留帧的patch token以供后续复用，
从而避免在VGGT的Aggregator中进行重复的特征编码计算。

典型用法:
    filter = KeyframeFilter(vggt_model.aggregator, threshold=0.35)
    result = filter(images)             # images: [B, S, 3, H, W]
    predictions = vggt_model(
        result["filtered_images"],
        precomputed_patch_tokens=result["patch_tokens"],
    )
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ImageNet 归一化常量，与 Aggregator 保持一致
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class KeyframeFilter:
    """
    基于DINOv2 CLS Token余弦相似度的视频关键帧前置过滤器。

    工作流程:
      1. 对输入帧进行与Aggregator相同的归一化处理
      2. 通过DINOv2 patch_embed 提取每帧的CLS Token和Patch Token
      3. 利用CLS Token的余弦距离进行关键帧筛选
      4. 缓存被保留帧的Patch Token，供VGGT直接复用

    参数:
        aggregator: VGGT的Aggregator模块（内含patch_embed即DINOv2编码器）
        threshold (float): 余弦距离阈值τ, D(f_t, f_ref) > τ 时判为关键帧
                          推荐范围: 0.2~0.5, 默认0.35（平衡模式，保留约30-40%帧）
        min_keyframes (int): 每个序列最少保留的关键帧数量
    """

    def __init__(
        self,
        aggregator,
        threshold: float = 0.35,
        min_keyframes: int = 2,
    ):
        self.patch_embed = aggregator.patch_embed
        self.threshold = threshold
        self.min_keyframes = min_keyframes

        # 获取 Aggregator 的归一化缓冲区（与其forward保持一致）
        self._resnet_mean = aggregator._resnet_mean
        self._resnet_std = aggregator._resnet_std

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入图像执行DINOv2编码，返回CLS Token和Patch Token。

        使用与Aggregator.forward完全相同的归一化流程，保证特征一致性。

        参数:
            images: [B, S, 3, H, W] 输入图像, 值域 [0, 1]

        返回:
            cls_tokens:   [B, S, embed_dim]  每帧的全局语义特征向量
            patch_tokens: [B*S, P, embed_dim] 每帧的空间特征图（可直接喂入Aggregator）
        """
        B, S, C, H, W = images.shape

        # 与 Aggregator.forward 第232-235行完全一致的归一化
        images_norm = images.to(torch.bfloat16)
        images_norm = (images_norm - self._resnet_mean) / self._resnet_std
        images_flat = images_norm.view(B * S, C, H, W)

        # 调用DINOv2编码器
        features = self.patch_embed(images_flat)

        if isinstance(features, dict):
            # DINOv2 ViT 模型返回 dict
            cls_tokens = features["x_norm_clstoken"]       # [B*S, embed_dim]
            patch_tokens = features["x_norm_patchtokens"]   # [B*S, P, embed_dim]
        else:
            # 简单 PatchEmbed (conv) 没有 cls_token
            patch_tokens = features  # [B*S, P, embed_dim]
            cls_tokens = features.mean(dim=1)  # 用 patch 均值近似全局特征

        cls_tokens = cls_tokens.view(B, S, -1)
        patch_tokens = patch_tokens.to(torch.bfloat16)

        return cls_tokens, patch_tokens

    @torch.no_grad()
    def select_keyframes(
        self, cls_tokens: torch.Tensor
    ) -> List[List[int]]:
        """
        基于CLS Token余弦距离的贪心关键帧选择。

        对每个batch独立执行：始终保留首帧作为参考帧，后续每帧与最近
        被选中的关键帧计算余弦距离 D = 1 - cos_sim，当 D > τ 时
        判定为新关键帧并更新参考帧。

        参数:
            cls_tokens: [B, S, embed_dim] 全局特征向量

        返回:
            keyframe_indices: List[List[int]], 每个batch元素的关键帧索引列表
        """
        B, S, _ = cls_tokens.shape

        # L2归一化以便计算余弦相似度
        cls_norm = F.normalize(cls_tokens.float(), dim=2)  # [B, S, D]

        all_indices = []
        for b in range(B):
            indices = [0]  # 始终保留第一帧
            ref_idx = 0

            for t in range(1, S):
                # 计算当前帧与参考帧的余弦距离
                cos_sim = (cls_norm[b, t] * cls_norm[b, ref_idx]).sum().item()
                distance = 1.0 - cos_sim

                if distance > self.threshold:
                    indices.append(t)
                    ref_idx = t  # 更新参考帧

            # 确保至少保留 min_keyframes 帧
            if len(indices) < self.min_keyframes and S >= self.min_keyframes:
                # 补充距离最大的帧
                remaining = sorted(set(range(S)) - set(indices))
                if remaining:
                    dists = []
                    for i in remaining:
                        d = 1.0 - (cls_norm[b, i] * cls_norm[b, 0]).sum().item()
                        dists.append((d, i))
                    dists.sort(reverse=True)
                    for _, idx in dists[: self.min_keyframes - len(indices)]:
                        indices.append(idx)
                indices = sorted(indices)

            all_indices.append(indices)

        return all_indices

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor
    ) -> Dict[str, object]:
        """
        执行关键帧预过滤与特征缓存。

        参数:
            images: [B, S, 3, H, W] 完整视频帧序列

        返回:
            dict:
                filtered_images:  [B, K, 3, H, W]  仅关键帧的图像
                patch_tokens:     [B*K, P, C]       预计算的patch token（可直接复用）
                keyframe_indices: List[List[int]]   每个batch的关键帧索引
                cls_tokens:       [B, S, C]         所有帧的CLS Token（供分析使用）
                stats: dict                         统计信息
        """
        B, S, C, H, W = images.shape

        # 第一步: DINOv2特征提取（所有帧）
        cls_tokens, all_patch_tokens = self.extract_features(images)
        # all_patch_tokens: [B*S, P, embed_dim]

        P = all_patch_tokens.shape[1]
        embed_dim = all_patch_tokens.shape[2]

        # 第二步: 关键帧选择
        keyframe_indices = self.select_keyframes(cls_tokens)

        # 第三步: 从缓存的patch token中按索引提取关键帧token
        # 将 patch_tokens 重塑为 [B, S, P, C] 以便按帧索引
        all_patch_tokens_reshaped = all_patch_tokens.view(B, S, P, embed_dim)

        filtered_images_list = []
        filtered_tokens_list = []
        max_K = max(len(idx) for idx in keyframe_indices)

        for b in range(B):
            idx = keyframe_indices[b]
            K = len(idx)
            idx_tensor = torch.tensor(idx, device=images.device)

            # 提取关键帧图像
            fi = images[b, idx_tensor]  # [K, 3, H, W]
            # 提取关键帧 patch tokens（零开销复用）
            ft = all_patch_tokens_reshaped[b, idx_tensor]  # [K, P, C]

            # 如需padding到统一长度（为batch对齐）
            if K < max_K:
                pad_imgs = fi[-1:].expand(max_K - K, -1, -1, -1)
                fi = torch.cat([fi, pad_imgs], dim=0)
                pad_toks = ft[-1:].expand(max_K - K, -1, -1)
                ft = torch.cat([ft, pad_toks], dim=0)

            filtered_images_list.append(fi)
            filtered_tokens_list.append(ft)

        filtered_images = torch.stack(filtered_images_list)   # [B, K, 3, H, W]
        filtered_tokens = torch.cat(
            [t.unsqueeze(0) for t in filtered_tokens_list], dim=0
        )  # [B, K, P, C]
        K = filtered_tokens.shape[1]
        # 重塑为 Aggregator 期望的格式: [B*K, P, C]
        precomputed_patch_tokens = filtered_tokens.view(B * K, P, embed_dim)

        # 统计信息
        total_frames = S
        kept_frames = max_K
        compression_ratio = kept_frames / total_frames

        return {
            "filtered_images": filtered_images,
            "patch_tokens": precomputed_patch_tokens,
            "keyframe_indices": keyframe_indices,
            "cls_tokens": cls_tokens,
            "stats": {
                "total_frames": total_frames,
                "kept_frames": kept_frames,
                "compression_ratio": compression_ratio,
                "threshold": self.threshold,
            },
        }
