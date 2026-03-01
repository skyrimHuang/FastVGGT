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
    def extract_single_frame_features(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单帧图像执行DINOv2编码，返回CLS Token和Patch Token。

        参数:
            frame: [B, 1, 3, H, W] 或 [B, 3, H, W] 单帧图像, 值域 [0, 1]

        返回:
            cls_token:   [B, embed_dim]  该帧的全局语义特征向量
            patch_token: [B, P, embed_dim] 该帧的空间特征图
        """
        if frame.ndim == 5:
            B, S, C, H, W = frame.shape
            assert S == 1, "extract_single_frame_features only accepts single frame"
            frame = frame.squeeze(1)  # [B, 3, H, W]
        
        B, C, H, W = frame.shape
        
        # 归一化
        frame_norm = frame.float()
        # _resnet_mean/std 在 aggregator 中是 [1,1,3,1,1]，需要 reshape 为 [1,3,1,1] 匹配 [B,3,H,W]
        resnet_mean = self._resnet_mean.float().to(frame.device).view(1, 3, 1, 1)
        resnet_std = self._resnet_std.float().to(frame.device).view(1, 3, 1, 1)
        frame_norm = (frame_norm - resnet_mean) / resnet_std
        
        # 调用DINOv2编码器
        features = self.patch_embed(frame_norm)
        
        if isinstance(features, dict):
            cls_token = features["x_norm_clstoken"]       # [B, embed_dim]
            patch_token = features["x_norm_patchtokens"]   # [B, P, embed_dim]
        else:
            patch_token = features  # [B, P, embed_dim]
            cls_token = features.mean(dim=1)  # 用 patch 均值近似全局特征
        
        return cls_token.float(), patch_token.float()

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor
    ) -> Dict[str, object]:
        """
        执行在线关键帧过滤与特征缓存（逐帧处理，避免OOM）。

        参数:
            images: [B, S, 3, H, W] 完整视频帧序列

        返回:
            dict:
                filtered_images:  [B, K, 3, H, W]  仅关键帧的图像
                patch_tokens:     [B*K, P, C]       预计算的patch token（可直接复用）
                keyframe_indices: List[List[int]]   每个batch的关键帧索引
                cls_tokens:       List[torch.Tensor] 所有帧的CLS Token（供分析使用）
                stats: dict                         统计信息
        """
        B, S, C, H, W = images.shape
        device = images.device
        
        # 逐batch处理（通常B=1）
        all_filtered_images = []
        all_filtered_tokens = []
        all_keyframe_indices = []
        all_cls_tokens = []
        
        for b in range(B):
            batch_images = images[b:b+1]  # [1, S, 3, H, W]
            
            # 在线过滤：逐帧提取+判断
            keyframe_indices = []
            filtered_images = []
            filtered_tokens = []
            cls_tokens_list = []
            
            ref_cls = None
            ref_idx = None
            
            for t in range(S):
                frame = batch_images[:, t:t+1]  # [1, 1, 3, H, W]
                
                # 提取特征
                cls_token, patch_token = self.extract_single_frame_features(frame)
                cls_tokens_list.append(cls_token)
                
                # 判断是否为关键帧
                if t == 0:
                    # 第一帧总是保留
                    is_keyframe = True
                else:
                    # 计算与参考帧的余弦距离
                    cls_norm = F.normalize(cls_token.float(), dim=-1)
                    ref_norm = F.normalize(ref_cls.float(), dim=-1)
                    cos_sim = (cls_norm * ref_norm).sum().item()
                    distance = 1.0 - cos_sim
                    is_keyframe = distance > self.threshold
                
                if is_keyframe:
                    keyframe_indices.append(t)
                    filtered_images.append(frame.squeeze(1))  # [1, 3, H, W]
                    filtered_tokens.append(patch_token)  # [1, P, C]
                    ref_cls = cls_token
                    ref_idx = t
            
            # 确保至少保留 min_keyframes 帧
            if len(keyframe_indices) < self.min_keyframes and S >= self.min_keyframes:
                remaining = sorted(set(range(S)) - set(keyframe_indices))
                if remaining:
                    # 计算剩余帧与第一帧的距离
                    first_cls_norm = F.normalize(cls_tokens_list[0].float(), dim=-1)
                    dists = []
                    for i in remaining:
                        cls_norm = F.normalize(cls_tokens_list[i].float(), dim=-1)
                        d = 1.0 - (cls_norm * first_cls_norm).sum().item()
                        dists.append((d, i))
                    dists.sort(reverse=True)
                    
                    # 补充距离最大的帧
                    for _, idx in dists[: self.min_keyframes - len(keyframe_indices)]:
                        # 重新提取该帧
                        frame = batch_images[:, idx:idx+1]
                        _, patch_token = self.extract_single_frame_features(frame)
                        keyframe_indices.append(idx)
                        filtered_images.append(frame.squeeze(1))
                        filtered_tokens.append(patch_token)
                    
                    keyframe_indices = sorted(keyframe_indices)
                    # 重新排序 filtered_images 和 filtered_tokens
                    idx_order = sorted(range(len(keyframe_indices)), key=lambda i: keyframe_indices[i])
                    filtered_images = [filtered_images[i] for i in idx_order]
                    filtered_tokens = [filtered_tokens[i] for i in idx_order]
            
            all_keyframe_indices.append(keyframe_indices)
            all_filtered_images.append(torch.cat(filtered_images, dim=0))  # [K, 3, H, W]
            all_filtered_tokens.append(torch.cat(filtered_tokens, dim=0))  # [K, P, C]
            all_cls_tokens.append(torch.cat(cls_tokens_list, dim=0))  # [S, C]
        
        # 统一batch的维度（padding到最大K）
        max_K = max(len(idx) for idx in all_keyframe_indices)
        
        padded_images = []
        padded_tokens = []
        
        for b in range(B):
            K = len(all_keyframe_indices[b])
            imgs = all_filtered_images[b]  # [K, 3, H, W]
            toks = all_filtered_tokens[b]  # [K, P, C]
            
            if K < max_K:
                # Padding
                pad_imgs = imgs[-1:].expand(max_K - K, -1, -1, -1)
                imgs = torch.cat([imgs, pad_imgs], dim=0)
                pad_toks = toks[-1:].expand(max_K - K, -1, -1)
                toks = torch.cat([toks, pad_toks], dim=0)
            
            padded_images.append(imgs)
            padded_tokens.append(toks)
        
        filtered_images = torch.stack(padded_images)  # [B, K, 3, H, W]
        filtered_tokens = torch.stack(padded_tokens)  # [B, K, P, C]
        
        # 重塑为 Aggregator 期望的格式
        K = filtered_tokens.shape[1]
        P = filtered_tokens.shape[2]
        C = filtered_tokens.shape[3]
        precomputed_patch_tokens = filtered_tokens.view(B * K, P, C)
        
        # 统计信息
        total_frames = S
        kept_frames = max_K
        compression_ratio = kept_frames / total_frames
        
        return {
            "filtered_images": filtered_images,
            "patch_tokens": precomputed_patch_tokens,
            "keyframe_indices": all_keyframe_indices,
            "cls_tokens": all_cls_tokens,  # List[Tensor]
            "stats": {
                "total_frames": total_frames,
                "kept_frames": kept_frames,
                "compression_ratio": compression_ratio,
                "threshold": self.threshold,
            },
        }

