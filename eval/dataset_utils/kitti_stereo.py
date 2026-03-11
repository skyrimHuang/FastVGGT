"""KITTI Scene Flow立体数据集加载器

带自动分辨率调整和calibration缩放的数据集实现
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from torchvision import transforms as TF

from .kitti_calib import KITTICalibrationProcessor, load_kitti_calibration


class KITTIStereoDataset:
    """
    KITTI Scene Flow立体数据集
    
    特点：
    - 自动将1242×375缩放到518×392（保持纵横比）
    - 自动缩放calibration参数
    - 计算GT scale标签（基于disparity和calibration）
    - 返回标准化的数据格式
    """
    
    # 标准KITTI原始分辨率
    ORIGINAL_SIZE = (1242, 375)
    # VGGT优化分辨率
    TARGET_SIZE = (518, 392)
    # 数据集样本数
    NUM_SAMPLES = 200
    
    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        enable_calib_cache: bool = True,
        resize_interpolation: str = 'cubic'
    ):
        """
        初始化KITTI数据集
        
        Args:
            data_dir: KITTI数据集根目录
            indices: 要使用的样本索引，默认使用全部(0-199)
            enable_calib_cache: 是否缓存calibration数据
            resize_interpolation: 图像插值方式，'cubic'或'linear'
        """
        self.data_dir = Path(data_dir)
        self.scene_flow_dir = self.data_dir / "data_scene_flow" / "training"
        self.calib_dir = self.data_dir / "data_scene_flow_calib" / "training" / "calib_cam_to_cam"
        
        # 验证目录存在
        if not self.scene_flow_dir.exists():
            raise FileNotFoundError(f"Scene flow data directory not found: {self.scene_flow_dir}")
        if not self.calib_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")
        
        # 样本索引
        self.indices = indices if indices is not None else list(range(self.NUM_SAMPLES))
        
        # Calibration缓存
        self.calib_cache = {} if enable_calib_cache else None
        self.processor = KITTICalibrationProcessor()
        
        # 插值方式
        self.interpolation = cv2.INTER_CUBIC if resize_interpolation == 'cubic' else cv2.INTER_LINEAR
        
        # 计算缩放因子（只计算一次）
        self.scale_x = self.TARGET_SIZE[0] / self.ORIGINAL_SIZE[0]
        self.scale_y = self.TARGET_SIZE[1] / self.ORIGINAL_SIZE[1]
    
    def _load_and_cache_calib(self, img_id: int) -> Dict:
        """
        加载并缓存calibration
        
        Args:
            img_id: 图像索引 (0-199)
        
        Returns:
            dict 包含calibration信息
        """
        # 检查缓存
        if self.calib_cache is not None and img_id in self.calib_cache:
            return self.calib_cache[img_id]
        
        # 加载calibration文件
        calib_path = self.calib_dir / f"{img_id:06d}.txt"
        calib_info = self.processor.parse_calib_file(calib_path)
        
        # 缩放K矩阵到目标分辨率
        K_scaled = self.processor.scale_intrinsics(
            calib_info['K_00'],
            self.ORIGINAL_SIZE,
            self.TARGET_SIZE
        )
        
        result = {
            'K_original': calib_info['K_00'],
            'K_scaled': K_scaled,
            'baseline': float(calib_info['baseline']),
            'scale_x': float(self.scale_x),
            'scale_y': float(self.scale_y)
        }
        
        # 保存到缓存
        if self.calib_cache is not None:
            self.calib_cache[img_id] = result
        
        return result
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Args:
            idx: 数据集中的索引 (0 to len(self)-1)
        
        Returns:
            dict 包含：
                - 'images': [2, 3, H, W] torch.Tensor 左右图像
                - 'disparity': [H, W] numpy.ndarray GT disparity
                - 'calibration': dict calibration参数
                - 'gt_scale': float GT尺度标签
                - 'metadata': dict 元数据
        
        Raises:
            IOError: 如果文件加载失败
        """
        img_id = self.indices[idx]
        
        # ========== 步骤1：加载左右图像 ==========
        img_left_path = self.scene_flow_dir / "image_2" / f"{img_id:06d}_10.png"
        img_right_path = self.scene_flow_dir / "image_3" / f"{img_id:06d}_10.png"
        
        try:
            img_left = cv2.imread(str(img_left_path))
            img_right = cv2.imread(str(img_right_path))
            
            if img_left is None or img_right is None:
                raise IOError(f"Failed to load images for sample {img_id:06d}")
            
            # 转换BGR -> RGB
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise IOError(f"Error loading images for sample {img_id:06d}: {e}")
        
        # ========== 步骤2：加载GT disparity ==========
        disp_path = self.scene_flow_dir / "disp_occ_0" / f"{img_id:06d}_10.png"
        
        try:
            # KITTI disparity以uint16格式存储，需要除以256才能得到实际值
            # 使用cv2.IMREAD_ANYDEPTH读取任意深度图像（包括uint16）
            disp_gt = cv2.imread(str(disp_path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
            
            if disp_gt is None:
                raise IOError(f"Failed to load disparity for sample {img_id:06d}")
        except Exception as e:
            raise IOError(f"Error loading disparity for sample {img_id:06d}: {e}")
        
        # ========== 步骤3：加载和缩放calibration ==========
        calib = self._load_and_cache_calib(img_id)
        
        # ========== 步骤4：调整图像和disparity到目标分辨率 ==========
        img_left_resized = cv2.resize(
            img_left,
            self.TARGET_SIZE,
            interpolation=self.interpolation
        )
        img_right_resized = cv2.resize(
            img_right,
            self.TARGET_SIZE,
            interpolation=self.interpolation
        )
        
        # Disparity resize（使用INTER_LINEAR保持，因为deep learning通常对差值不敏感）
        disp_resized = cv2.resize(
            disp_gt,
            self.TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )
        
        # ========== 步骤5：计算GT scale标签 ==========
        # gt_scale = baseline * focal_length / median_valid_disparity (改用median，更robust)
        # 其中focal_length是缩放后的值，确保与网络输出空间一致
        valid_mask = disp_resized > 0
        
        if valid_mask.sum() > 0:
            median_disp = np.median(disp_resized[valid_mask])  # 改用median替代mean
            focal_length = calib['K_scaled'][0, 0]
            gt_scale = calib['baseline'] * focal_length / median_disp
        else:
            # 如果没有有效的disparity像素，使用默认值
            gt_scale = 1.0
        
        # ========== 步骤6：转换为tensor ==========
        to_tensor = TF.ToTensor()
        img_left_tensor = to_tensor(Image.fromarray(img_left_resized))
        img_right_tensor = to_tensor(Image.fromarray(img_right_resized))
        images_stereo = torch.stack([img_left_tensor, img_right_tensor])  # [2, 3, H, W]
        
        # ========== 步骤7：返回数据字典 ==========
        return {
            'images': images_stereo,  # [2, 3, 518, 392]
            'disparity': disp_resized,  # [518, 392]
            'calibration': calib,
            'gt_scale': torch.tensor(gt_scale, dtype=torch.float32),
            'metadata': {
                'img_id': img_id,
                'img_id_str': f"{img_id:06d}",
                'size_original': self.ORIGINAL_SIZE,
                'size_resized': self.TARGET_SIZE
            }
        }


def kitti_collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，处理KITTI batch的特殊情况
    
    Args:
        batch: list of dict 从__getitem__返回的数据
    
    Returns:
        dict 堆叠后的batch数据
    """
    batch_dict = {
        'images': torch.stack([b['images'] for b in batch]),
        'disparity': np.array([b['disparity'] for b in batch]),  # [B, H, W]
        'gt_scale': torch.stack([b['gt_scale'] for b in batch]),  # [B]
    }
    
    # 堆叠calibration（需要特殊处理字典）
    calib_list = [b['calibration'] for b in batch]
    batch_dict['calibration'] = {
        'K_original': np.array([c['K_original'] for c in calib_list]),  # [B, 3, 3]
        'K_scaled': np.array([c['K_scaled'] for c in calib_list]),      # [B, 3, 3]
        'baseline': np.array([c['baseline'] for c in calib_list]),      # [B]
        'scale_x': calib_list[0]['scale_x'],  # 所有样本相同
        'scale_y': calib_list[0]['scale_y']
    }
    
    # 元数据
    batch_dict['metadata'] = [b['metadata'] for b in batch]
    
    return batch_dict
