"""KITTI立体尺度预测头模块

用于预测monocular深度的尺度因子，使其达到真实尺度
"""

import torch
import torch.nn as nn
from typing import Optional


class KITTIStereoScaleHead(nn.Module):
    """
    KITTI立体尺度预测头
    
    输入：
    - 左右帧的聚合tokens（来自VGGT aggregator最后一层）
    - Calibration参数（baseline、焦距等）
    
    输出：
    - 尺度因子（应乘以monocular深度得到metric深度）
    
    架构：
    - Token平均池化把 [B, 2, P, C] 转换为 [B, 2, C]
    - 左右特征连接：[B, 2*C]
    - 融合calibration信息：[B, 2*C+2]
    - 3层MLP + LayerNorm + GELU dropout
    - 输出log_scale（确保输出为正）
    """
    
    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_calibration_features: bool = True
    ):
        """
        初始化ScaleHead
        
        Args:
            dim_in: 输入token维度，通常为2*embed_dim（左右帧合并）
            hidden_dims: 隐层维度列表，默认[1024, 512]
            dropout: Dropout比例
            use_calibration_features: 是否使用calibration特征
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        
        self.dim_in = dim_in
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_calibration_features = use_calibration_features
        
        # 计算MLP输入维度
        # 左右特征拼接 + calibration特征（baseline, focal_scaled）
        calib_feat_dim = 2 if use_calibration_features else 0
        mlp_in_dim = dim_in * 2 + calib_feat_dim
        self.mlp_in_dim = mlp_in_dim
        
        # 构建MLP层
        layers = []
        prev_dim = mlp_in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层：预测log_scale（单个标量）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        tokens_last_layer: torch.Tensor,
        calibration_features: Optional[torch.Tensor] = None,
        patch_start_idx: int = 0
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens_last_layer: [B, 2, P, C]
                B: batch size
                2: 左右帧
                P: patch数量（包含 camera + register + patch tokens）
                C: token维度
            
            calibration_features: [B, 2] (可选)
                包含：[baseline, focal_length_scaled]
                若为None且use_calibration_features=True，则不使用calibration
            
            patch_start_idx: int
                patch tokens 的起始索引（跳过 camera + register tokens）
                与 DPTHead / TrackHead 保持一致
        
        Returns:
            scale: [B, 1]
                尺度因子，范围 > 0
        """
        B = tokens_last_layer.shape[0]
        
        # ========== 提取 patch tokens（排除 camera + register tokens）==========
        # 与 DPTHead 保持一致：aggregated_tokens_list[i][:, :, patch_start_idx:]
        patch_tokens = tokens_last_layer[:, :, patch_start_idx:]  # [B, 2, P_patch, C]
        
        # ========== Token平均池化 ==========
        # 在patch维度（dim=2）进行平均，把 [B, 2, P_patch, C] -> [B, 2, C]
        x_pooled = patch_tokens.mean(dim=2)  # [B, 2, C]
        
        # 分离左右特征
        x_left = x_pooled[:, 0, :]   # [B, C]
        x_right = x_pooled[:, 1, :]  # [B, C]
        
        # ========== 特征连接 ==========
        x = torch.cat([x_left, x_right], dim=1)  # [B, 2*C]
        
        # ========== 融合calibration信息（可选）==========
        if self.use_calibration_features and calibration_features is not None:
            # calibration_features: [B, 2]
            x = torch.cat([x, calibration_features], dim=1)  # [B, 2*C+2]

        if x.shape[1] != self.mlp_in_dim:
            raise RuntimeError(
                f"ScaleHead input dim mismatch: got {x.shape[1]}, expected {self.mlp_in_dim}. "
                f"tokens_last_layer shape={tuple(tokens_last_layer.shape)}, "
                f"calibration_features shape={None if calibration_features is None else tuple(calibration_features.shape)}"
            )
        
        # ========== MLP ==========
        log_scale = self.mlp(x)  # [B, 1]
        
        # ========== 指数映射确保正值 ==========
        scale = torch.exp(log_scale)  # [B, 1]
        
        # 同时保存 log_scale 供 loss 直接使用（避免 exp→log 往返的数值误差）
        self._last_log_scale = log_scale
        
        return scale


class SimpleStereoScaleHead(nn.Module):
    """
    简化版的立体尺度预测头（不依赖calibration特征）
    
    用于测试或轻量级应用
    """
    
    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dims: list = None,
        dropout: float = 0.1
    ):
        """初始化简化版ScaleHead"""
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        
        # MLP：[2*C] -> scale
        layers = []
        prev_dim = dim_in * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, tokens_last_layer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens_last_layer: [B, 2, P, C]
        
        Returns:
            scale: [B, 1]
        """
        # Token平均池化
        x_pooled = tokens_last_layer.mean(dim=2)  # [B, 2, C]
        
        # 左右特征连接
        x_left = x_pooled[:, 0, :]
        x_right = x_pooled[:, 1, :]
        x = torch.cat([x_left, x_right], dim=1)  # [B, 2*C]
        
        # MLP
        log_scale = self.mlp(x)
        scale = torch.exp(log_scale)
        
        return scale


# ==================== 工厂函数 ====================

def create_scale_head(
    scale_head_type: str = "kitti",
    dim_in: int = 2048,
    **kwargs
) -> nn.Module:
    """
    创建尺度预测头的工厂函数
    
    Args:
        scale_head_type: "kitti" 或 "simple"
        dim_in: 输入token维度
        **kwargs: 传递给corresponding模块的参数
    
    Returns:
        ScaleHead module
    """
    if scale_head_type == "kitti":
        return KITTIStereoScaleHead(dim_in=dim_in, **kwargs)
    elif scale_head_type == "simple":
        return SimpleStereoScaleHead(dim_in=dim_in, **kwargs)
    else:
        raise ValueError(f"Unknown scale head type: {scale_head_type}")
