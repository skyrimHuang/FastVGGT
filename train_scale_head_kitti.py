#!/usr/bin/env python3
"""
KITTI立体尺度预测头的训练脚本

用于训练VGGT模型的scale_head，使其能预测metric深度的尺度因子
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging
import time
from datetime import datetime
from collections import defaultdict

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vggt.models.vggt import VGGT
from eval.dataset_utils.kitti_stereo import KITTIStereoDataset, kitti_collate_fn


# ==================== 性能计时器 ====================

class Timer:
    """用于测量推理时间的计时器"""
    def __init__(self):
        self.times = []
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.times.append(time.time() - self.start_time)
    
    @property
    def avg(self) -> float:
        return np.mean(self.times) if self.times else 0.0
    
    @property
    def median(self) -> float:
        return np.median(self.times) if self.times else 0.0
    
    @property
    def fps(self) -> float:
        return 1.0 / self.avg if self.avg > 0 else 0.0
    
    def reset(self):
        self.times = []


# ==================== 训练历史记录 ====================

class TrainingHistory:
    """记录训练历史（Loss曲线、指标等）"""
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 历史数据
        self.history = defaultdict(list)
        
        # JSON文件路径
        self.json_path = self.output_dir / "training_history.json"
        self.csv_path = self.output_dir / "training_history.csv"
        
        # CSV写入器
        self.csv_file = None
        self.csv_writer = None
    
    def record_epoch(self, epoch: int, metrics: dict):
        """
        记录单个Epoch的指标
        
        Args:
            epoch: Epoch编号
            metrics: 指标字典，包括train_loss, val_mean_error, val_fps等
        """
        metrics['epoch'] = epoch
        
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def save(self, logger: logging.Logger = None):
        """
        保存训练历史到JSON和CSV文件
        
        Args:
            logger: 日志记录器
        """
        # 保存为JSON
        try:
            with open(self.json_path, 'w') as f:
                json.dump(dict(self.history), f, indent=2)
            if logger:
                logger.info(f"✓ Training history saved to {self.json_path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save JSON: {e}")
        
        # 保存为CSV
        try:
            if len(self.history) > 0:
                # 获取所有键
                keys = list(self.history.keys())
                epochs = self.history.get('epoch', [])
                
                if len(epochs) > 0:
                    with open(self.csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        
                        for i in range(len(epochs)):
                            row = {key: self.history[key][i] for key in keys}
                            writer.writerow(row)
                    
                    if logger:
                        logger.info(f"✓ Training history saved to {self.csv_path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save CSV: {e}")
    
    def get_history(self) -> dict:
        """获取历史数据"""
        return dict(self.history)
    
    def load(self, logger: logging.Logger = None) -> bool:
        """
        从文件加载训练历史
        
        Returns:
            bool: 是否加载成功
        """
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.history = defaultdict(list)
                    for key, values in data.items():
                        self.history[key] = values
                
                if logger:
                    logger.info(f"✓ Loaded training history from {self.json_path}")
                return True
            except Exception as e:
                if logger:
                    logger.error(f"Failed to load history: {e}")
                return False
        return False


# ==================== 日志配置 ====================

def setup_logging(log_dir: str):
    """设置日志"""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # 创建自定义日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(Path(log_dir) / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def print_training_header(logger: logging.Logger, config: dict):
    """打印训练头信息"""
    logger.info("=" * 80)
    logger.info(" " * 20 + "VGGT KITTI Stereo Scale Head Training")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("[Dataset Config]")
    logger.info(f"  Data Dir: {config['data']['data_dir']}")
    logger.info(f"  Train Samples: {config['data']['train_num']}")
    logger.info(f"  Val Samples: {config['data']['val_num']}")
    logger.info(f"  Batch Size: {config['data']['batch_size']}")
    logger.info(f"  Num Workers: {config['data']['num_workers']}")
    logger.info("")
    logger.info("[Model Config]")
    logger.info(f"  Image Size: {config['model'].get('img_size', 518)}")
    logger.info(f"  Patch Size: {config['model'].get('patch_size', 14)}")
    logger.info(f"  Embed Dim: {config['model'].get('embed_dim', 1024)}")
    logger.info(f"  Checkpoint: {config['model']['ckpt_path']}")
    logger.info("")
    logger.info("[Training Config]")
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Learning Rate: {config['training']['lr']}")
    logger.info(f"  Weight Decay: {config['training']['weight_decay']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Early Stopping Patience: {config['training']['early_stopping_patience']}")
    logger.info(f"  Save Interval: {config['training']['save_interval']}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")


# ==================== 损失函数 ====================

class ScaleHeadLoss(nn.Module):
    """尺度预测的组合损失函数"""
    
    def __init__(self, loss_weights: dict = None):
        """
        Args:
            loss_weights: 各损失项的权重
        """
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {'scale_loss': 1.0}
        
        self.loss_weights = loss_weights
    
    def forward(self, pred_scale: torch.Tensor, gt_scale: torch.Tensor,
                log_pred: torch.Tensor = None) -> dict:
        """
        计算损失
        
        Args:
            pred_scale: [B, 1] 预测的scale因子（exp后）
            gt_scale: [B] 或 [B, 1] GT scale因子
            log_pred: [B, 1] 可选，MLP直接输出的log_scale（避免exp→log往返）
        
        Returns:
            dict 包含各项损失
        """
        if gt_scale.dim() == 1:
            gt_scale = gt_scale.unsqueeze(1)
        
        # 主损失：对数空间MSE（更稳定）
        log_gt = torch.log(gt_scale + 1e-8)
        
        # 优先使用 MLP 直接输出的 log_scale，避免 exp→log 数值往返
        if log_pred is None:
            log_pred = torch.log(pred_scale + 1e-8)
        
        scale_loss = F.mse_loss(log_pred, log_gt)
        
        losses = {
            'scale_loss': scale_loss,
            'total_loss': scale_loss * self.loss_weights['scale_loss']
        }
        
        return losses


# ==================== 训练和验证函数 ====================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: ScaleHeadLoss,
    device: str,
    logger: logging.Logger,
    epoch: int
) -> dict:
    """
    单个epoch的训练
    
    Args:
        model: VGGT模型（scale_head已启用）
        train_loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 计算设备
        logger: 日志记录器
        epoch: epoch序号
    
    Returns:
        dict 包含平均损失等指标
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch:>3d}", ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device=device, dtype=torch.float32)  # [B, 2, 3, H, W] aggregator内部转bfloat16
        gt_scale = batch['gt_scale'].to(device)  # [B]
        
        # 提取calibration特征
        calib = batch['calibration']
        baseline = torch.tensor(calib['baseline']).to(device).float()
        focal_scaled = torch.tensor([c[0, 0] for c in calib['K_scaled']]).to(device).float()
        
        calib_features = torch.stack([baseline, focal_scaled], dim=1)  # [B, 2]
        
        # 前向传播
        with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
            predictions = model(images, calibration_batch=calib_features)
            
            if 'scale_factor' not in predictions:
                raise RuntimeError("scale_factor not found in predictions. "
                                   "Make sure scale_head is enabled and images have 2 frames.")
            
            pred_scale = predictions['scale_factor']  # [B, 1]
            
            # 获取MLP直接输出的log_scale（避免exp→log数值往返）
            log_pred = getattr(model.scale_head, '_last_log_scale', None)
            
            # 计算损失
            losses = loss_fn(pred_scale, gt_scale, log_pred=log_pred)
            loss = losses['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.scale_head.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss/num_batches:.6f}'
        })
    
    avg_loss = total_loss / num_batches
    
    return {'train_loss': avg_loss}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    logger: logging.Logger
) -> dict:
    """
    验证
    
    Args:
        model: VGGT模型
        val_loader: 验证数据加载器
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        dict 包含验证指标
    """
    model.eval()
    
    pred_scales = []
    gt_scales = []
    errors = []
    inference_times = []  # 推理时间
    
    pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['images'].to(device=device, dtype=torch.float32)  # [B, 2, 3, H, W] aggregator内部转bfloat16
            gt_scale = batch['gt_scale'].to(device).cpu().numpy()
            
            # 提取calibration特征
            calib = batch['calibration']
            baseline = torch.tensor(calib['baseline']).to(device).float()
            focal_scaled = torch.tensor([c[0, 0] for c in calib['K_scaled']]).to(device).float()
            calib_features = torch.stack([baseline, focal_scaled], dim=1)
            
            # 推理并计时
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
                predictions = model(images, calibration_batch=calib_features)
                pred_scale = predictions['scale_factor'].squeeze(1).cpu().numpy()  # [B]
            
            torch.cuda.synchronize() if device == 'cuda' else None
            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time / len(images))  # 单位：秒/样本
            
            pred_scales.extend(pred_scale)
            gt_scales.extend(gt_scale)
            
            # 计算relative error
            rel_error = np.abs(pred_scale - gt_scale) / (gt_scale + 1e-8)
            errors.extend(rel_error)
    
    pred_scales = np.array(pred_scales)
    gt_scales = np.array(gt_scales)
    errors = np.array(errors)
    inference_times = np.array(inference_times)
    
    # 计算统计指标
    mean_error = errors.mean()
    median_error = np.median(errors)
    rmse = np.sqrt(((pred_scales - gt_scales) ** 2).mean())
    mae = np.abs(pred_scales - gt_scales).mean()
    
    # Delta指标（绝对误差在阈值内的比例）
    delta_thresholds = [0.05, 0.1, 0.15]
    deltas = {}
    for threshold in delta_thresholds:
        deltas[f'delta_{threshold:.2f}'] = (errors < threshold).mean()
    
    # 推理性能
    avg_inference_time = inference_times.mean()
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # 输出详细信息
    logger.info("")
    logger.info("[Validation Results]")
    logger.info(f"  Mean Relative Error: {mean_error:.4f} ({mean_error*100:.2f}%)")
    logger.info(f"  Median Relative Error: {median_error:.4f} ({median_error*100:.2f}%)")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    for threshold, ratio in deltas.items():
        logger.info(f"  {threshold}: {ratio*100:.2f}%")
    logger.info(f"  Avg Inference Time: {avg_inference_time*1000:.2f}ms/sample")
    logger.info(f"  FPS: {fps:.2f}")
    
    return {
        'val_mean_error': mean_error,
        'val_median_error': median_error,
        'val_rmse': rmse,
        'val_mae': mae,
        'val_fps': fps,
        'val_inference_time': avg_inference_time,
        'pred_scale_mean': pred_scales.mean(),
        'gt_scale_mean': gt_scales.mean(),
        **deltas
    }


# ==================== 主训练循环 ====================

def train(config: dict, device: str):
    """
    主训练函数
    
    Args:
        config: 配置字典
        device: 计算设备
    """
    # 设置日志
    logger = setup_logging(config['training']['output_dir'])
    print_training_header(logger, config)
    
    # 创建输出目录
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    # ========== 数据加载 ==========
    logger.info("[Step 1] Loading datasets...")
    
    train_dataset = KITTIStereoDataset(
        data_dir=config['data']['data_dir'],
        indices=list(range(config['data']['train_num']))
    )
    
    val_dataset = KITTIStereoDataset(
        data_dir=config['data']['data_dir'],
        indices=list(range(config['data']['train_num'], config['data']['train_num'] + config['data']['val_num']))
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=kitti_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=kitti_collate_fn
    )
    
    logger.info(f"✓ Train dataset: {len(train_dataset)} samples")
    logger.info(f"✓ Val dataset: {len(val_dataset)} samples")
    logger.info(f"✓ Iterations per epoch: {len(train_loader)}")
    logger.info("")
    
    # ========== 模型加载 ==========
    logger.info("[Step 2] Loading model...")
    
    model = VGGT(
        img_size=config['model'].get('img_size', 518),
        patch_size=config['model'].get('patch_size', 14),
        embed_dim=config['model'].get('embed_dim', 1024),
        enable_camera=False,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_scale_head=True,
        merging=config['model'].get('merging', 0),
        merge_ratio=config['model'].get('merge_ratio', 0.9)
    )
    
    # 加载预训练权重
    ckpt_path = config['model']['ckpt_path']
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        incompat = model.load_state_dict(ckpt, strict=False)
        logger.info(f"✓ Missing keys: {len(incompat[0])}, Unexpected keys: {len(incompat[1])}")
    else:
        logger.warning(f"✗ Checkpoint not found: {ckpt_path}")
    
    model = model.to(device)
    logger.info(f"✓ Model loaded on {device}")
    logger.info("")
    
    # ========== 冻结backbone权重 ==========
    logger.info("[Step 3] Freezing backbone parameters...")
    
    if model.aggregator is not None:
        for param in model.aggregator.parameters():
            param.requires_grad = False
    if model.camera_head is not None:
        for param in model.camera_head.parameters():
            param.requires_grad = False
    if model.depth_head is not None:
        for param in model.depth_head.parameters():
            param.requires_grad = False
    if model.point_head is not None:
        for param in model.point_head.parameters():
            param.requires_grad = False
    if model.track_head is not None:
        for param in model.track_head.parameters():
            param.requires_grad = False
    
    # 计算可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Total parameters: {total_params:,}")
    logger.info(f"✓ Trainable parameters (ScaleHead only): {trainable_params:,}")
    logger.info("")
    
    # ========== 优化器和调度器 ==========
    logger.info("[Step 4] Setting up optimizer and scheduler...")
    
    optimizer = optim.AdamW(
        model.scale_head.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    loss_fn = ScaleHeadLoss(loss_weights=config['training']['loss_weights'])
    
    logger.info(f"✓ Optimizer: AdamW")
    logger.info(f"✓ Scheduler: CosineAnnealingLR (T_max={config['training']['epochs']})")
    logger.info("")
    
    # ========== 初始化训练历史 ==========
    logger.info("[Step 5] Initializing training history...")
    
    history_recorder = TrainingHistory(output_dir)
    start_epoch = 0
    
    # 如果指定了resume checkpoint，加载历史和优化器状态
    resume_path = config['training'].get('resume_ckpt_path', None)
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        
        # 加载scale_head权重
        scale_head_state = torch.load(resume_path, map_location=device)
        model.scale_head.load_state_dict(scale_head_state)
        logger.info(f"✓ Loaded scale_head weights")
        
        # 尝试加载训练历史
        if history_recorder.load(logger):
            start_epoch = len(history_recorder.history.get('epoch', []))
            logger.info(f"✓ Resuming from epoch {start_epoch}")
    
    logger.info(f"✓ Training history recorder initialized")
    logger.info("")
    
    # ========== 训练循环 ==========
    logger.info("[Step 6] Starting training loop...")
    logger.info("=" * 80)
    
    best_val_error = float('inf')
    patience_count = 0
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, logger, epoch
        )
        
        # 验证
        val_metrics = validate(model, val_loader, device, logger)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史 - 保存到TrainingHistory
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'val_mean_error': val_metrics['val_mean_error'],
            'val_median_error': val_metrics['val_median_error'],
            'val_rmse': val_metrics['val_rmse'],
            'val_mae': val_metrics['val_mae'],
            'val_fps': val_metrics['val_fps'],
            'val_inference_time': val_metrics['val_inference_time'],
            'learning_rate': current_lr
        }
        
        # 添加delta指标
        for key, value in val_metrics.items():
            if key.startswith('delta_'):
                epoch_metrics[key] = value
        
        history_recorder.record_epoch(epoch, epoch_metrics)
        
        epoch_time = time.time() - epoch_start
        
        # 打印综合结果
        logger.info(f"Epoch {epoch:>3d}/{config['training']['epochs']} | "
                   f"Loss: {train_metrics['train_loss']:.6f} | "
                   f"Val Error: {val_metrics['val_mean_error']*100:>6.2f}% | "
                   f"FPS: {val_metrics['val_fps']:>7.2f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        
        # 检查最优模型
        if val_metrics['val_mean_error'] < best_val_error:
            best_val_error = val_metrics['val_mean_error']
            patience_count = 0
            
            # 保存最优模型
            best_ckpt_path = ckpt_dir / 'scale_head_best.pt'
            torch.save(model.scale_head.state_dict(), best_ckpt_path)
            logger.info(f"  ★ Best model updated! Error: {best_val_error*100:.2f}%")
        else:
            patience_count += 1
            if patience_count > 0:
                logger.info(f"  ← No improvement ({patience_count}/{config['training']['early_stopping_patience']})")
        
        # Early stopping
        if patience_count >= config['training']['early_stopping_patience']:
            logger.info("")
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 定期保存checkpoints
        if (epoch + 1) % config['training']['save_interval'] == 0:
            ckpt_path = ckpt_dir / f'scale_head_epoch_{epoch:03d}.pt'
            torch.save(model.scale_head.state_dict(), ckpt_path)
            logger.info(f"  → Checkpoint saved: epoch_{epoch:03d}.pt")
        
        logger.info("-" * 80)
    
    total_time = time.time() - start_time
    
    # 保存训练历史
    history_recorder.save(logger)
    
    # 获取历史数据用于总结
    hist_data = history_recorder.get_history()
    
    # 最终总结
    logger.info("")
    logger.info("=" * 80)
    logger.info(" " * 25 + "Training Completed!")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_time/3600:.2f}h")
    logger.info(f"Best validation error: {best_val_error*100:.2f}%")
    logger.info(f"Best model saved to: {best_ckpt_path}")
    logger.info("")
    logger.info("[Final Metrics]")
    if len(hist_data['train_loss']) > 0:
        logger.info(f"  Final train loss: {hist_data['train_loss'][-1]:.6f}")
        logger.info(f"  Final val error: {hist_data['val_mean_error'][-1]*100:.2f}%")
        logger.info(f"  Final val rmse: {hist_data['val_rmse'][-1]:.6f}")
        logger.info(f"  Best val error: {min(hist_data['val_mean_error'])*100:.2f}%")
        logger.info(f"  Average FPS: {np.mean(hist_data['val_fps']):.2f}")
    logger.info("=" * 80)


# ==================== 主入口 ====================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train KITTI stereo scale head for VGGT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认配置训练
  python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml
  
  # 覆盖配置参数
  python train_scale_head_kitti.py \\
    --config configs/train_scale_head_kitti.yaml \\
    --epochs 50 \\
    --batch_size 16 \\
    --lr 5e-4 \\
    --output_dir outputs/scale_head_v1
  
  # 使用GPU指定训练
  python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml --device cuda:0
        """
    )
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_scale_head_kitti.yaml',
        help='Path to config file (default: configs/train_scale_head_kitti.yaml)'
    )
    
    # 设备选择
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: cuda, cuda:0, cuda:1, cpu (default: cuda)'
    )
    
    # 数据相关
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Override config: path to KITTI dataset'
    )
    parser.add_argument(
        '--train_num',
        type=int,
        help='Override config: number of training samples'
    )
    parser.add_argument(
        '--val_num',
        type=int,
        help='Override config: number of validation samples'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Override config: batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        help='Override config: number of data loading workers'
    )
    
    # 模型相关
    parser.add_argument(
        '--ckpt_path',
        type=str,
        help='Override config: path to pretrained checkpoint'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        help='Override config: embedding dimension'
    )
    
    # 训练相关
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override config: number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Override config: learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Override config: weight decay'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        help='Override config: early stopping patience'
    )
    
    # 输出相关
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Override config: output directory for checkpoints and logs'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖命令行参数
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.train_num:
        config['data']['train_num'] = args.train_num
    if args.val_num:
        config['data']['val_num'] = args.val_num
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    
    if args.ckpt_path:
        config['model']['ckpt_path'] = args.ckpt_path
    if args.embed_dim:
        config['model']['embed_dim'] = args.embed_dim
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['lr'] = args.lr
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    if args.early_stopping_patience:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # 检查CUDA可用性
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Device: {args.device}")
    print(f"Config file: {args.config}")
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"{'='*60}\n")
    
    # 运行训练
    train(config, args.device)


if __name__ == "__main__":
    main()
