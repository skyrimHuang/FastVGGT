# 关键帧过滤与特征复用评估 —— 完整指南

## 概述

本文档介绍如何在真实数据集（7Scenes、ScanNet等）上全面评估**基于DINOv2特征的关键帧预过滤与特征复用机制**的性能、可靠性和实用性。

### 核心设计

```
输入视频 [B, S, 3, H, W]
    ↓
[关键帧过滤模块] DINOv2特征提取 + 余弦相似度判别
    ↓
保留关键帧 + 缓存patches [B, K, ...] + patch_tokens [B*K, P, C]
    ↓
[VGGT推理] 使用预计算patch_tokens，0开销复用
    ↓
深度/相机/追踪预测
```

## 快速开始

### 1. 快速验证（无需真实数据）

```bash
cd /home/hba/Documents/FastVGGT_2

# 运行验证脚本，测试所有核心功能（合成数据，CPU/GPU都可）
python tests/test_eval_pipeline.py
```

**输出示例：**
```
✓ 特征提取: [1, 10, 3, 518, 518] → CLS [1, 10, 64]
✓ 关键帧选择: τ=0.35, 2/20 帧保留 (10%)
✓ 完整管道: 2个batch × 15帧序列正常处理
✓ 余弦距离分析: 平均距离0.0000
✅ 所有验证测试通过！
```

### 2. 真实数据评估

#### 前置条件

- 数据集已下载：7Scenes 或 ScanNet
- 模型检查点：`/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt`
- GPU显存 ≥ 12GB（推荐16GB以上）

#### 使用7Scenes数据集

```bash
python tests/eval_keyframe_filter_realdata.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --sequence_lengths 5,10,15,20,30 \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --num_samples 3 \
  --output_dir ./tests/eval_output_7scenes
```

#### 使用ScanNet数据集

```bash
python tests/eval_keyframe_filter_realdata.py \
  --dataset_type scannet \
  --data_dir /home/hba/Documents/Dataset/ScanNet/scans \
  --sequence_lengths 5,10,15,20,30 \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --num_samples 5 \
  --output_dir ./tests/eval_output_scannet
```

### 3. 输出结果

所有结果保存在 `--output_dir` 中：

```
output_dir/
├── eval_keyframe_filter_detailed.csv      # 详细结果表（每个配置）
├── eval_keyframe_filter_summary.csv       # 汇总表（按序列长度）
├── eval_report.txt                       # 文本报告
├── fig_cosine_distance_dist.png          # 余弦距离分布（直方图）
├── fig_threshold_vs_retention.png        # 阈值vs保留帧率（曲线）
├── fig_timing_speedup.png                # 推理时间 & 加速比（柱&线图）
└── fig_memory_vs_sequence.png            # 显存vs序列长度（曲线）
```

## 评估维度详解

### 1️⃣ **特征区分度** (Feature Discriminability)

**什么是区分度？**
- DINOv2 CLS token在不同视频帧中的差异大小
- 通过**余弦距离** D = 1 - cos_sim 衡量
- 距离越大 → 帧越不同 → 越容易被过滤器区分

**评估方法：**
- 提取所有帧的DINOv2 CLS token
- 计算相邻帧的余弦距离
- 绘制分布直方图（按序列长度）

**图表：** `fig_cosine_distance_dist.png`
- 6个子图，分别对应6个序列长度
- 显示余弦距离分布是否符合期望（平滑视频距离小，快速变化距离大）

**预期结果：**
- 平滑运动：余弦距离 0.01~0.05（框架易于选择）
- 快速运动：余弦距离 0.1~0.3（区分度高）
- 随机帧：余弦距离 0.2~0.5（高区分度）

---

### 2️⃣ **关键帧压缩率** (Compression Ratio)

**定义：** 保留帧数 / 总帧数

**评估方法：**
- 对不同阈值τ（0.1~0.7），统计保留的关键帧比例
- 曲线越陡 → 阈值的可调节性越好

**图表：** `fig_threshold_vs_retention.png`
- X轴：阈值τ
- Y轴：保留帧率（%）
- 多条线：不同序列长度

**预期结果：**
```
τ=0.1  → 100% 保留（几乎所有帧都很不同）
τ=0.35 → 30~40% 保留（推荐平衡点）
τ=0.7  → <10% 保留（极度过滤）
```

---

### 3️⃣ **推理时间与加速比** (Timing & Speedup)

**评估方法：**
1. **无过滤推理时间** T₁ = VGGT(原始视频)
2. **过滤+复用推理时间** T₂ = 过滤时间 + VGGT(过滤视频 + 复用tokens)
3. **加速比** = T₁ / T₂

**图表：**  `fig_timing_speedup.png`
- 左图：时间对比（柱状图）
  - 蓝：无过滤
  - 橙：过滤+复用
- 右图：加速倍数（曲线图）

**预期结果：**
```
序列长度  5  →  加速比 1.1×  (过滤开销 > 收益)
序列长度 20  →  加速比 1.5×  (平衡)
序列长度 30  →  加速比 2.0×  (过滤收益明显)
```

---

### 4️⃣ **显存使用与OOM边界** (Memory & OOM)

**评估方法：**
1. 逐步增加序列长度（5, 10, 15, 20, 30）
2. 监测GPU峰值显存占用
3. 记录OOM时的序列长度

**图表：** `fig_memory_vs_sequence.png`
- X轴：序列长度
- Y轴：峰值显存 (MB)
- 填充曲线：显示内存上升趋势

**预期结果：**
```
线性增长或接近线性，如：
  5帧  → 2GB
  10帧 → 3GB
  20帧 → 5GB
  30帧 → 7GB
  40帧 → OOM (取决于GPU)
```

**OOM扫描方法：**
脚本会自动：
1. 尝试当前序列长度
2. 捕获 `torch.cuda.OutOfMemoryError`
3. 清理显存继续下一个配置
4. 记录最大可处理的序列长度

---

### 5️⃣ **特征复用一致性** (Reuse Consistency)

**评估方法：**
- 计算两条推理路径的预测差异
  - 路径1: 直接推理所有帧 → 深度图 D₁
  - 路径2: 过滤后复用tokens → 深度图 D₂
- 比较 D₁[关键帧位置] 与 D₂ 的差异

**一致性指标：**
```
MSE = mean((D1[keyframes] - D2)²)
```

**预期结果：**
```
MSE < 0.01  → 完美复用（数值精度一致）
MSE < 0.1   → 优秀复用（可忽略的误差）
MSE > 1.0   → 复用有问题
```

---

## 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_type` | `7scenes` | 数据集类型：7scenes / scannet |
| `--data_dir` | `/home/hba/Documents/Dataset/7_scenes` | 数据集根目录 |
| `--sequence_lengths` | `5,10,15,20,30` | 要测试的序列长度 |
| `--thresholds` | `0.1,0.2,0.3,0.35,0.5,0.7` | 关键帧过滤阈值 |
| `--num_samples` | `3` | 每个配置测试的样本数（多样性） |
| `--ckpt_path` | 默认模型路径 | 模型检查点 |
| `--device` | `cuda:0` | 推理设备 |
| `--seed` | `42` | 随机种子（可重复性） |

## CSV输出格式

### `eval_keyframe_filter_detailed.csv`

```csv
sequence_length,threshold,compression_ratio,time_no_filter,time_filter,time_inference_filtered,time_total,speedup,peak_memory_mb
5,0.1,1.0,0.123,0.001,0.120,0.121,1.017,2048
5,0.2,0.8,0.123,0.001,0.095,0.096,1.281,1928
5,0.35,0.6,0.123,0.001,0.071,0.072,1.708,1824
...
```

**列含义：**
- `sequence_length`: 输入序列帧数
- `threshold`: 关键帧过滤阈值τ
- `compression_ratio`: 保留帧数/总帧数
- `time_no_filter`: 无过滤推理时间(秒)
- `time_filter`: 关键帧过滤耗时(秒)
- `time_inference_filtered`: 过滤后推理耗时(秒)
- `time_total`: 总耗时(秒)
- `speedup`: 加速倍数
- `peak_memory_mb`: 峰值显存(MB)

### `eval_keyframe_filter_summary.csv`

```csv
sequence_length,compression_ratio,speedup,peak_memory_mb
5,0.65,1.34,2128.3
10,0.52,1.68,2945.2
15,0.48,1.92,3821.5
20,0.45,2.15,4756.1
30,0.42,2.31,6542.8
```

## 故障排除

### ❌ OOM太早（序列长度<10就OOM）

**原因：**
1. GPU显存不足（需要 ≥ 8GB）
2. 后台进程占用显存

**解决：**
```bash
# 清空GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 检查GPU使用
nvidia-smi

# 减少样本数量
--num_samples 1
```

### ❌ 加速比 < 1（反而变慢）

**原因：**
1. 过滤开销 > 推理节省时间
2. 序列长度太短（< 5帧）

**解决：**
```bash
# 测试更长的序列
--sequence_lengths 10,20,30,40,50

# 使用更激进的阈值（过滤更多帧）
--thresholds 0.5,0.7
```

### ❌ CSV中出现NaN或Inf

**原因：** 推理失败或显存溢出

**解决：**
- 检查 `eval_report.txt` 中的错误信息
- 减少 `--num_samples`
- 使用较小的 `--sequence_lengths`

## 预期输出示例

### 终端输出

```
======================================================================
关键帧预过滤与特征复用 —— 真实数据评估
======================================================================

[1/5] 加载模型...
✓ 加载检查点: /home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt

[2/5] 数据加载与特征区分度评估...
  [████████████████] 5/5 序列长度

[3/5] 生成中文图表...
✓ 保存余弦距离分布图: ./tests/eval_output/fig_cosine_distance_dist.png
✓ 保存阈值vs保留帧率图: ./tests/eval_output/fig_threshold_vs_retention.png
✓ 保存时间加速图: ./tests/eval_output/fig_timing_speedup.png
✓ 保存显存使用图: ./tests/eval_output/fig_memory_vs_sequence.png

[4/5] 保存CSV结果表...
✓ 详细结果: ./tests/eval_output/eval_keyframe_filter_detailed.csv
✓ 汇总表: ./tests/eval_output/eval_keyframe_filter_summary.csv

[5/5] 生成评估报告...

======================================================================
评估结果汇总
======================================================================
数据集: 7scenes
样本数: 3
序列长度: [5, 10, 15, 20, 30]
阈值范围: 0.10 - 0.70

主要指标:
  平均压缩率: 58.3%
  平均加速比: 1.73×
  峰值显存: 6542 MB
  最长序列: 30帧
======================================================================
✅ 评估完成！输出目录: ./tests/eval_output_7scenes
```

## 进阶用法

### 自定义阈值扫描以找到最优点

```bash
# 精细化扫描：0.25~0.45，步长0.05
python tests/eval_keyframe_filter_realdata.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --sequence_lengths 20 \
  --thresholds 0.25,0.3,0.35,0.4,0.45 \
  --num_samples 5
```

然后查看 `fig_threshold_vs_retention.png`，找到加速比最高且压缩率可接受的阈值。

### 测试OOM边界

```bash
# 逐步增加序列长度至OOM
python tests/eval_keyframe_filter_realdata.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --sequence_lengths 10,15,20,25,30,35,40,45,50 \
  --thresholds 0.35 \
  --num_samples 1
```

脚本会自动记录每个序列长度的最大显存和OOM边界。

### 比较不同数据集

```bash
# 7Scenes评估
python tests/eval_keyframe_filter_realdata.py --dataset_type 7scenes --output_dir eval_7scenes

# ScanNet评估
python tests/eval_keyframe_filter_realdata.py --dataset_type scannet --output_dir eval_scannet

# 结果自动保存在不同目录，便于对比
```

## 论文相关表格和图

本评估脚本生成的所有CSV和图表均可直接用于论文补充材料：

| 内容 | 文件 | 推荐用途 |
|------|------|---------|
| 余弦距离分布 | `fig_cosine_distance_dist.png` | 论文 Figure：特征区分度分析 |
| 阈值vs压缩率 | `fig_threshold_vs_retention.png` | 论文 Figure：关键帧选择参数敏感性 |
| 时间加速比 | `fig_timing_speedup.png` | 论文 Figure：提出方法的加速效果 |
| 显存使用 | `fig_memory_vs_sequence.png` | 论文 Table/Figure：可扩展性分析 |
| 详细数据 | `eval_keyframe_filter_detailed.csv` | 论文 Appendix：定量结果 |

## 参考文献

- DINOv2: "Emerging Properties in Self-Supervised Vision Transformers" (Meta AI)
- VGGT: 本项目提出的视觉几何跟踪网络

## 常见问题

**Q: 为什么要在关键帧过滤前进行特征提取？**
A: 为了在VGGT推理前及早筛选框架，避免对冗余帧进行繁重的编码。

**Q: 特征复用是否会影响最终的深度/追踪精度？**
A: 否。复用只跳过了DINOv2编码器，后续的Aggregator、Heads等计算完全相同。

**Q: 如何选择最优的阈值τ？**
A: 根据应用场景：
- 实时应用：τ=0.5~0.7（极度压缩，加速比2~3×）
- 高精度应用：τ=0.2~0.3（温和压缩，加速比1.1~1.3×）
- 平衡模式：τ=0.35（推荐，30~40%压缩，加速比1.5~1.8×）

## 致谢

感谢7Scenes、ScanNet数据集维护者提供的公开数据。

---

**更新时间：** 2026年2月28日  
**作者：** FastVGGT Evaluation Team
