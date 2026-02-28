"""KITTI Calibration预处理工具

处理KITTI camera calibration到VGGT目标分辨率的缩放
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class KITTICalibrationProcessor:
    """处理KITTI calibration文件和参数缩放"""
    
    # 标准KITTI原始分辨率（scene flow任务中的标准）
    KITTI_ORIGINAL_SIZE = (1242, 375)
    
    # VGGT优化分辨率（保持纵横比，确保是14的倍数）
    VGGT_TARGET_SIZE = (518, 392)
    
    @staticmethod
    def parse_calib_file(calib_path: str) -> Dict[str, np.ndarray]:
        """
        解析KITTI calibration文件
        
        Args:
            calib_path: 指向calib_XXXXXX.txt文件的路径
        
        Returns:
            dict包含：
                - K_00: [3, 3] 左图内参矩阵
                - baseline: float 立体基线（米）
                - calib_raw: dict 原始所有参数
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 解析失败
        """
        calib_dict = {}
        calib_path = Path(calib_path)
        
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        try:
            with open(calib_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(':', 1)
                    if len(parts) != 2:
                        continue
                    
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # 解析为浮点数数组
                    try:
                        values = np.array([float(x) for x in value_str.split()])
                        calib_dict[key] = values
                    except ValueError:
                        continue
        except Exception as e:
            raise ValueError(f"Failed to parse calibration file {calib_path}: {e}")
        
        # 提取关键参数
        if 'P_rect_00' not in calib_dict:
            raise ValueError("P_rect_00 not found in calibration file")
        
        if 'T_01' not in calib_dict:
            raise ValueError("T_01 (baseline) not found in calibration file")
        
        # P_rect_00 = K @ [R|t]，提取K矩阵
        P_rect_00 = calib_dict['P_rect_00'].reshape(3, 4)
        K_00 = P_rect_00[:, :3]
        
        # 从 T_01 向量计算baseline（左到右的距离）
        baseline_vec = calib_dict['T_01']
        baseline = np.linalg.norm(baseline_vec)
        
        return {
            'K_00': K_00,
            'baseline': baseline,
            'calib_raw': calib_dict
        }
    
    @staticmethod
    def scale_intrinsics(
        K: np.ndarray,
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        缩放内参矩阵
        
        Args:
            K: [3, 3] 原始K矩阵
            orig_size: (width, height) 原始KITTI大小，通常(1242, 375)
            target_size: (width, height) 新目标大小，通常(518, 392)
        
        Returns:
            K_scaled: [3, 3] 缩放后的K矩阵
        
        公式：
            fx_new = fx_old × (target_w / orig_w)
            fy_new = fy_old × (target_h / orig_h)
            cx_new = cx_old × (target_w / orig_w)
            cy_new = cy_old × (target_h / orig_h)
        """
        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        
        scale_x = target_w / orig_w  # 通常 0.4173
        scale_y = target_h / orig_h  # 通常 1.0453
        
        K_scaled = K.copy().astype(np.float32)
        K_scaled[0, 0] *= scale_x   # fx
        K_scaled[1, 1] *= scale_y   # fy
        K_scaled[0, 2] *= scale_x   # cx（主点也需要缩放）
        K_scaled[1, 2] *= scale_y   # cy
        
        return K_scaled
    
    @staticmethod
    def compute_target_resolution(
        orig_w: int = 1242,
        orig_h: int = 375,
        target_w: int = 518
    ) -> Tuple[int, int]:
        """
        自动计算目标分辨率
        
        保持纵横比，确保是14的倍数（VGGT patch size）
        
        Args:
            orig_w: 原始宽度
            orig_h: 原始高度
            target_w: 目标宽度
        
        Returns:
            (target_w, target_h) 满足14倍数的分辨率
        """
        scale = target_w / orig_w
        target_h = int(np.ceil(orig_h * scale / 14) * 14)
        
        # 上限限制（避免过度拉伸）
        if target_h > 518:
            target_h = (518 // 14) * 14
        
        return target_w, target_h
    
    @classmethod
    def compute_scale_factors(
        cls,
        orig_size: Tuple[int, int] = None,
        target_size: Tuple[int, int] = None
    ) -> Tuple[float, float]:
        """
        计算缩放因子
        
        Args:
            orig_size: 原始大小，若为None则使用标准值
            target_size: 目标大小，若为None则使用标准值
        
        Returns:
            (scale_x, scale_y)
        """
        if orig_size is None:
            orig_size = cls.KITTI_ORIGINAL_SIZE
        if target_size is None:
            target_size = cls.VGGT_TARGET_SIZE
        
        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        return scale_x, scale_y
    
    @staticmethod
    def verify_calibration_scaling(
        K_orig: np.ndarray,
        K_scaled: np.ndarray,
        scale_x: float,
        scale_y: float,
        tolerance: float = 1e-5
    ) -> bool:
        """
        验证calibration缩放的正确性
        
        Args:
            K_orig: 原始K矩阵
            K_scaled: 缩放后K矩阵
            scale_x: 预期的x轴缩放因子
            scale_y: 预期的y轴缩放因子
            tolerance: 允许的误差
        
        Returns:
            bool 缩放是否正确
        """
        # 验证焦距缩放
        actual_scale_x = K_scaled[0, 0] / K_orig[0, 0]
        actual_scale_y = K_scaled[1, 1] / K_orig[1, 1]
        
        fx_ok = abs(actual_scale_x - scale_x) < tolerance
        fy_ok = abs(actual_scale_y - scale_y) < tolerance
        
        # 验证主点缩放
        cx_actual_scale = K_scaled[0, 2] / K_orig[0, 2]
        cy_actual_scale = K_scaled[1, 2] / K_orig[1, 2]
        
        cx_ok = abs(cx_actual_scale - scale_x) < tolerance
        cy_ok = abs(cy_actual_scale - scale_y) < tolerance
        
        return fx_ok and fy_ok and cx_ok and cy_ok


def load_kitti_calibration(
    calib_path: str,
    target_size: Tuple[int, int] = None
) -> Dict:
    """
    便捷函数：加载并缩放KITTI calibration
    
    Args:
        calib_path: calibration文件路径
        target_size: 目标分辨率，若为None则使用默认(518, 392)
    
    Returns:
        dict 包含：
            - K_original: 原始K矩阵
            - K_scaled: 缩放后K矩阵
            - baseline: 立体基线
            - scale_x, scale_y: 缩放因子
    """
    if target_size is None:
        target_size = KITTICalibrationProcessor.VGGT_TARGET_SIZE
    
    processor = KITTICalibrationProcessor()
    
    # 解析calibration文件
    calib_data = processor.parse_calib_file(calib_path)
    K_orig = calib_data['K_00']
    baseline = calib_data['baseline']
    
    # 缩放K矩阵
    K_scaled = processor.scale_intrinsics(
        K_orig,
        processor.KITTI_ORIGINAL_SIZE,
        target_size
    )
    
    scale_x, scale_y = processor.compute_scale_factors(
        processor.KITTI_ORIGINAL_SIZE,
        target_size
    )
    
    return {
        'K_original': K_orig,
        'K_scaled': K_scaled,
        'baseline': baseline,
        'scale_x': scale_x,
        'scale_y': scale_y
    }
