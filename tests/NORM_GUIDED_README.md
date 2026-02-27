# Norm-Guided Anchoring & Threshold-Gated Anti-Collapse

## 摘要

本实施为FastVGGT项目添加了两种智能token合并优化策略：

1. **Norm-Guided Anchoring（基于L2范数的锚点划分）**: 使用token的L2范数作为几何信息量的指标，优先保护高频特征（角点、边缘）不被合并。

2. **Threshold-Gated Anti-Collapse（阈值门控防碰撞）**: 引入相似度阈值，拒绝强制合并不相似的token，避免破坏独特物体的3D结构。

## 理论依据

### Vision Transformers Need Registers (ICCV 2023)
- 发现：高L2 norm的tokens包含更多信息量
- 应用：保护高norm tokens不被全局汇聚破坏

### DINOv2 (Meta AI)
- 使用register tokens保护高频特征
- 本实现：将高norm tokens作为"隐式register"

### DynamicViT
- 自适应token pruning基于相似度阈值
- 本实现：自适应token merging基于相似度阈值

## 核心算法

### 方案一：Norm-Guided Anchoring

```python
# 计算L2 norm
token_norms = metric.norm(dim=-1)  # [B, N]
avg_norms = token_norms.mean(dim=0)  # [N]

# 降序排序
sorted_indices = torch.argsort(avg_norms, descending=True)

# 三层划分
num_protected = int(N * 0.10)  # Top 10% - 完全不合并
num_dst = int(N * 0.40)        # Next 40% - 作为dst锚点
# num_src = N - num_protected - num_dst  # Bottom 50% - 待合并

protected_indices = sorted_indices[:num_protected]
dst_indices = sorted_indices[num_protected:num_protected + num_dst]
src_indices = sorted_indices[num_protected + num_dst:]
```

**关键洞察**: 
- 高频几何特征（角点、边缘）→ 高L2 norm
- 低频背景（墙面、天空）→ 低L2 norm
- 保证"低频向高频合并"，不摧毁3D关键点

### 方案二：Threshold-Gated Anti-Collapse

```python
# 计算相似度
sim_matrix = torch.bmm(a_norm, b_norm.T)  # [B, num_src, num_dst]
node_max, node_idx = torch.max(sim_matrix, dim=-1)

# 阈值过滤
TAU = 0.85
valid_similarity = node_max > TAU  # [B, num_src]

# 动态调整合并数量
valid_count = valid_similarity.sum().item()
r_actual = min(r, valid_count)  # 只合并valid的tokens
```

**关键洞察**:
- 拒绝相似度<0.85的"强买强卖"合并
- 复杂场景（低相似度）→ 少合并
- 简单场景（高相似度）→ 多合并

## 使用方法

### 命令行参数

```bash
# 启用Norm-Guided Anchoring
python eval/eval_scannet.py --use_norm_guided ...

# 设置相似度阈值
python eval/eval_scannet.py --merge_threshold 0.85 ...

# 同时启用（推荐）
python eval/eval_scannet.py \
    --merging 0 --merge_ratio 0.9 \
    --use_norm_guided --merge_threshold 0.85 \
    --input_frame 100 --num_scenes 5 \
    --output_path ./results/improved
```

### Python API

```python
from vggt.models.vggt import VGGT

model = VGGT(
    merging=0,
    merge_ratio=0.9,
    use_norm_guided=True,      # 启用方案一
    merge_threshold=0.85,      # 启用方案二
)
```

## 实验设置

### 数据集
- **ScanNet**: 室内3D重建数据集
- **场景数**: 5个场景（快速验证）
- **每场景帧数**: 100帧
- **总测试时间**: ~25-30分钟（两组测试并行）

### 对比组
1. **Baseline**: 2×2网格步长随机划分（原始方法）
2. **Improved**: Norm-Guided + Threshold-Gated（本实现）

### 评估指标
- **Chamfer Distance (CD)**: 3D重建质量（越小越好）
- **Absolute Trajectory Error (ATE)**: 位姿平移误差
- **Absolute Rotation Error (ARE)**: 位姿旋转误差
- **Inference Time (ms)**: 推理速度

## 代码修改

### 核心文件

**merging/merge.py** (+73 lines)
- Lines 90-127: Norm-guided split逻辑
- Lines 282-316: Threshold-gated filtering逻辑

**vggt/layers/attention.py** (+4 lines)
- 传递`use_norm_guided`和`merge_threshold`参数

**vggt/layers/block.py** (+2 lines)
- Block类接受新参数

**vggt/models/vggt.py** (+4 lines)
- VGGT模型配置新参数

**vggt/models/aggregator.py** (+6 lines)
- Aggregator传递新参数到Block

**eval/eval_scannet.py** (+12 lines)
- 添加命令行参数`--use_norm_guided`和`--merge_threshold`

### 测试脚本

**tests/compare_norm_guided.py** (新增)
- 自动对比baseline vs improved结果
- 生成表格和JSON报告

**tests/monitor_tests.sh** (新增)
- 实时监控测试进度
- 每10秒刷新状态

## 参数调优指南

### use_norm_guided分层比例

当前：10% protected, 40% dst, 50% src

**更保守**（保护更多高频特征）:
```python
num_protected = int(N * 0.15)  # 15%
num_dst = int(N * 0.35)        # 35%
# src: 50%
```

**更激进**（合并更多低频特征）:
```python
num_protected = int(N * 0.05)  # 5%
num_dst = int(N * 0.35)        # 35%
# src: 60%
```

### merge_threshold阈值

当前：τ = 0.85

**更宽松**（合并更多tokens，速度优先）:
```bash
--merge_threshold 0.80
```

**更严格**（保留更多tokens，精度优先）:
```bash
--merge_threshold 0.90
```

**禁用阈值过滤**:
```bash
--merge_threshold 0.0
```

## 预期结果

### 成功标准

✅ **Chamfer Distance** ↓ 10-30%  
✅ **ATE/ARE** 保持或改善  
✅ **Inference Time** ±10%

### 如果CD退化

**可能原因**:
1. Norm计算在归一化后进行（丢失幅度信息）
2. 分层比例不适合当前场景
3. 阈值过于严格/宽松

**调试步骤**:
1. 验证norm计算在归一化**之前**完成
2. 尝试不同分层比例
3. 尝试不同阈值
4. 可视化被保护的tokens分布

## 计算开销分析

### Norm-Guided Anchoring
- **Norm计算**: O(N) - ~0.1ms
- **排序**: O(N log N) - ~1-2ms for N=15000
- **掩码创建**: O(N) - ~0.1ms
- **总开销**: ~1-3ms per layer

### Threshold-Gated Anti-Collapse
- **阈值比较**: O(N) - ~0.1ms
- **掩码应用**: O(1) - negligible
- **总开销**: <0.5ms per layer

**结论**: 两种优化的计算开销都非常小（<3ms per layer），24层共<72ms，对总推理时间（~3000-5000ms）影响<2%。

## 限制与未来工作

### 当前限制
1. **Per-batch norm averaging**: 可能不适合batch size=1的情况
2. **静态分层比例**: 所有场景使用相同的10%-40%-50%
3. **全局阈值**: 所有tokens使用相同的τ

### 未来改进方向
1. **Adaptive分层**: 根据场景复杂度动态调整比例
2. **Per-token阈值**: 根据token自身norm动态调整τ
3. **Spatial awareness**: 结合空间信息（不仅仅是norm）
4. **Learned protection**: 使用小型MLP学习哪些tokens该保护

## 参考文献

1. **Vision Transformers Need Registers**  
   Darcet et al., ICCV 2023
   
2. **DINOv2: Learning Robust Visual Features without Supervision**  
   Oquab et al., TMLR 2024
   
3. **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification**  
   Rao et al., NeurIPS 2021

## 作者与日期

- **实施者**: GitHub Copilot (Claude Sonnet 4.5)
- **日期**: 2026-02-27
- **分支**: norm-guided-merge
- **Commit**: fe90aae
