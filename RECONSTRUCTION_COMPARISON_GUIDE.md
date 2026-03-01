# 过滤前后重建精度对比评估指南

## 📋 概述

`eval/eval_reconstruction_comparison.py` 是一个专门用于对比过滤前后重建精度的评估脚本。它支持 **7Scenes** 和 **ScanNet** 两种主流数据集，分别评估：

1. **无过滤方案**（Baseline）：直接使用所有输入帧进行推理和重建
2. **有过滤方案**（过滤后）：使用 keyframe_filter 过滤冗余帧后进行推理和重建
3. **OOM 优雅处理**：当显存溢出时，不中断执行，记录为 "OOM"

## 🚀 快速开始

### 1. 7Scenes 评估（推荐首先尝试）

```bash
# 快速测试（1个场景，5帧）
python eval/eval_reconstruction_comparison.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --scene_names chess \
  --thresholds 0.3,0.5 \
  --input_frame 5 \
  --output_dir ./tests/eval_recon_quick

# 完整评估（所有场景）
python eval/eval_reconstruction_comparison.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --scene_names chess,fire,heads,office,pumpkin,redkitchen,stairs \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --input_frame 20 \
  --output_dir ./tests/eval_recon_7scenes
```

### 2. ScanNet 评估

```bash
# 快速测试（5个场景）
python eval/eval_reconstruction_comparison.py \
  --dataset_type scannet \
  --data_dir /home/hba/Documents/Dataset/ScanNet/scans \
  --num_scenes 5 \
  --thresholds 0.3,0.5 \
  --input_frame 20 \
  --output_dir ./tests/eval_recon_scannet_quick

# 完整评估（50个场景）
python eval/eval_reconstruction_comparison.py \
  --dataset_type scannet \
  --data_dir /home/hba/Documents/Dataset/ScanNet/scans \
  --num_scenes 50 \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --input_frame 100 \
  --output_dir ./tests/eval_recon_scannet_full
```

## 📊 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset_type` | 数据集类型 | `7scenes` 或 `scannet` |
| `--data_dir` | 数据集根目录 | `/home/hba/Documents/Dataset/7_scenes` |

### 过滤参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--thresholds` | `0.3,0.5,0.7` | 过滤阈值（余弦距离），逗号分隔 |

**阈值说明：**
- **τ = 0.1～0.2**：保留帧数多（~70%），精度最高，计算量大
- **τ = 0.3～0.35**：平衡模式（~30-40%），推荐使用
- **τ = 0.5～0.7**：激进模式（~15-20%），计算快但可能丢失细节

### 数据集特定参数

#### 7Scenes
| 参数 | 说明 | 示例 |
|------|------|------|
| `--scene_names` | 场景列表（逗号分隔），不指定则默认使用所有场景 | `chess,fire,heads,office` |

可用场景：`chess`, `fire`, `heads`, `office`, `pumpkin`, `redkitchen`, `stairs`

#### ScanNet
| 参数 | 说明 | 示例 |
|------|------|------|
| `--num_scenes` | 最多评估的场景数量 | `10` |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_frame` | `100` | 每个场景/序列的最大帧数 |
| `--depth_conf_thresh` | `1.0` | 深度置信度阈值（过滤低置信度深度值） |
| `--chamfer_max_dist` | `0.5` | Chamfer 距离最大值（单位：米） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt_path` | 见脚本 | 模型检查点路径 |
| `--merging` | `None` | 是否使用 token merging |
| `--merge_ratio` | `0.9` | Token merge 比例 |

### 系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cuda:0` | 计算设备 |
| `--output_dir` | `./tests/eval_reconstruction` | 输出目录 |

## 📈 输出说明

评估完成后会生成：

### 1. CSV 结果文件 (`reconstruction_comparison.csv`)

```
dataset,scene,method,threshold,frames,success,inference_time_ms,memory_mb,compression_ratio,error,kept_frames
7scenes,chess,no_filter,,5,True,1041.30,3164.4,1.0,,
7scenes,chess,filter,0.3,5,True,303.85,2751.1,0.4,,2.0
7scenes,chess,filter,0.5,5,True,232.97,2751.1,0.4,,2.0
```

**字段说明：**
- `dataset`: 数据集类型
- `scene`: 场景名称
- `method`: 推理方法 (`no_filter` 或 `filter`)
- `threshold`: 过滤阈值（`no_filter` 时为空）
- `frames`: 输入总帧数
- `success`: 是否成功 (True/False)
- `inference_time_ms`: 推理时间（毫秒）
- `memory_mb`: 峰值显存占用（MB）
- `compression_ratio`: 压缩率 (kept_frames / input_frames)
- `error`: 错误信息（如 "OOM"，成功时为空）
- `kept_frames`: 保留的帧数（仅 filter 方法有）

### 2. 控制台输出

```
📊 评估摘要:
   总行数: 3
   成功: 3
   失败: 0

   按方法分类:
     no_filter   :   1/  1 成功
     filter      :   2/  2 成功

⚡ 性能统计（仅成功案例）:
   无过滤:
     平均推理时间: 1041.30 ms
     平均显存占用: 3164 MB
   有过滤 (平均):
     平均推理时间: 268.41 ms
     平均显存占用: 2751 MB
     平均压缩率: 40.0%
     平均保留帧数: 2.0

   相对改进:
     推理加速: 3.88x
     显存降低: 13.1%
```

## 🧪 典型评估场景

### 场景 A：了解过滤效果（推荐）
```bash
# 目标：快速了解过滤方案的整体效果
# 耗时：~5 分钟

python eval/eval_reconstruction_comparison.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --scene_names chess,fire \
  --thresholds 0.3,0.5 \
  --input_frame 10 \
  --output_dir ./tests/eval_recon_demo
```

### 场景 B：论文级评估
```bash
# 目标：生成论文中使用的完整评估结果
# 耗时：~30-60 分钟

python eval/eval_reconstruction_comparison.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --scene_names chess,fire,heads,office,pumpkin,redkitchen,stairs \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --input_frame 50 \
  --output_dir ./tests/eval_recon_paper_7scenes

# 同时评估 ScanNet
python eval/eval_reconstruction_comparison.py \
  --dataset_type scannet \
  --data_dir /home/hba/Documents/Dataset/ScanNet/scans \
  --num_scenes 50 \
  --thresholds 0.1,0.2,0.3,0.35,0.5,0.7 \
  --input_frame 100 \
  --output_dir ./tests/eval_recon_paper_scannet
```

### 场景 C：OOM 边界测试
```bash
# 目标：找到 OOM 边界，验证过滤方案的关键价值
# 耗时：~1-2 小时

python eval/eval_reconstruction_comparison.py \
  --dataset_type 7scenes \
  --data_dir /home/hba/Documents/Dataset/7_scenes \
  --scene_names office \
  --thresholds 0.3,0.5,0.7 \
  --input_frame 200 \
  --output_dir ./tests/eval_recon_oom_boundary
```

## 💡 使用技巧

### 1. 快速定位问题
如果某些配置失败，检查 CSV 中的 `error` 列：
- `OOM`: 显存溢出，需要调整阈值或减少帧数
- `FilterError: ...`: 过滤器出错，通常是数据格式问题
- `InferenceError: ...`: 推理出错，通常是模型权重问题

### 2. 找到最优过滤阈值
在 CSV 中对比不同阈值的 `inference_time_ms` 和 `compression_ratio`：
```python
import pandas as pd
df = pd.read_csv('reconstruction_comparison.csv')

# 查看不同阈值的性能对比
filter_results = df[df['method'] == 'filter']
for threshold in filter_results['threshold'].unique():
    subset = filter_results[filter_results['threshold'] == threshold]
    avg_time = subset['inference_time_ms'].mean()
    avg_ratio = subset['compression_ratio'].mean()
    print(f"τ={threshold}: 时间={avg_time:.1f}ms, 压缩率={avg_ratio:.1%}")
```

### 3. 处理 OOM 情况
如果发生 OOM：
1. 减少 `--input_frame` 参数
2. 增加过滤阈值（例如 0.7）
3. 检查系统其他程序是否占用显存

### 4. 生成论文图表
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('reconstruction_comparison.csv')

# 图1：推理时间对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 无过滤 vs 有过滤
no_filter = df[df['method'] == 'no_filter'].groupby('scene')['inference_time_ms'].mean()
filter_data = df[df['method'] == 'filter'].groupby('scene')['inference_time_ms'].mean()

ax = axes[0]
x = range(len(no_filter))
ax.bar([i - 0.2 for i in x], no_filter, width=0.4, label='No Filter')
ax.bar([i + 0.2 for i in x], filter_data, width=0.4, label='Filtered')
ax.set_xlabel('Scene')
ax.set_ylabel('Inference Time (ms)')
ax.legend()

# 图2：压缩率
compression = df[df['method'] == 'filter'].groupby('threshold')['compression_ratio'].mean()
axes[1].plot(compression.index, compression.values * 100, marker='o')
axes[1].set_xlabel('Filter Threshold (τ)')
axes[1].set_ylabel('Compression Ratio (%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=300)
```

## 📝 输出日志解读

运行时的日志格式示例：

```
📍 处理场景: chess / seq-01
    输入: 5 帧, 分辨率 torch.Size([5, 3, 392, 518])
    1️⃣  无过滤推理... ✓ (时间: 1041.30ms, 显存: 3164MB)
    2️⃣  过滤推理 (τ=0.3)... ✓ (保留帧: 2/5, 时间: 303.85ms, 显存: 2751MB)
    2️⃣  过滤推理 (τ=0.5)... ✓ (保留帧: 2/5, 时间: 232.97ms, 显存: 2751MB)
```

**关键信息：**
- ✓ 成功，❌ 失败
- 保留帧数（对于理解压缩效果）
- 推理时间（越低越好）
- 显存占用（越低越好）

## 🔍 故障排除

### Q1: 报错 "无效的场景数据"
**原因**：7Scenes 的数据格式问题
**解决**：检查 `--data_dir/scene_name/seq-01/` 下是否有 `.color.png` 和对应的 `.pose.txt` 文件

### Q2: 过滤推理出现 dtype 错误
**原因**：已修复，不应该在新版本中出现
**解决**：更新到最新版本脚本

### Q3: OOM 发生
**原因**：显存不足
**解决**：
- 减少帧数：`--input_frame 10`
- 增加过滤阈值：`--thresholds 0.7`
- 关闭其他程序释放显存

### Q4: CSV 某些行全是 None（OOM 情况）
**说明**：正常行为！脚本优雅地处理了 OOM，继续处理其他配置
**利用**：分析 OOM 的发生条件（场景、阈值、帧数）

## 📚 相关文献参考

在论文中引用此评估时：

```
我们使用 VGGT 原生框架的重建评估工具对比了过滤前后的重建质量。
评估在 7Scenes 和 ScanNet 数据集上进行，分别考察无过滤方案（直接使用所有帧）
和使用动态关键帧过滤的方案（τ ∈ {0.1, 0.2, 0.3, 0.35, 0.5, 0.7}）。
结果表明，过滤方案在保持重建精度的同时，实现了平均 X 倍的加速和 Y% 的显存节省。
```

## ✅ 验证清单

运行评估前检查：

- [ ] Python 环境已激活：`conda activate fastvggt`
- [ ] CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 数据集路径正确：`ls $data_dir | head`
- [ ] 输出目录有写权限：`touch $output_dir/test.txt`
- [ ] 模型检查点存在：`ls -lh $ckpt_path`

运行评估后检查：

- [ ] CSV 文件已生成：`ls -lh $output_dir/*.csv`
- [ ] 数据行数合理：行数应为 (1 + len(thresholds)) × num_scenes
- [ ] 成功率大于 50%：表示评估基本可行
- [ ] 推理时间合理：单帧推理应在 100-2000ms

## 📞 问题反馈

如遇到问题，请检查：
1. 脚本日志中的完整错误信息
2. CSV 中的 `error` 列
3. GPU 显存使用情况：`nvidia-smi`
4. 数据集文件完整性

---

**最后更新**: 2026-03-01  
**脚本版本**: 1.0  
**支持数据集**: 7Scenes, ScanNet
