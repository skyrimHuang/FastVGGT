# Norm-Guided Anchoring & Threshold-Gated Anti-Collapse Implementation

## 实施状态

### ✅ 完成的工作

#### 1. 代码实现
- **方案一：Norm-Guided Anchoring** (Lines 90-127 in merge.py)
  - 基于L2 norm对tokens进行降序排序
  - 三层划分：10% protected (完全不合并), 40% dst (锚点), 50% src (待合并)
  - 替换原有的2×2网格步长随机划分机制
  
- **方案二：Threshold-Gated Anti-Collapse** (Lines 282-316 in merge.py)
  - 添加相似度阈值过滤 (默认τ=0.85)
  - 拒绝强制合并低相似度token对
  - 动态调整实际合并数量r_actual
  
- **参数传递链路**:
  - eval_scannet.py → VGGT → Aggregator → Block → Attention → token_merge_bipartite2d
  - 新增参数：`--use_norm_guided`, `--merge_threshold`

#### 2. 已修改的文件
```
merging/merge.py                 (+73 lines) - 核心算法实现
vggt/layers/attention.py         (+4 lines)  - 参数传递
vggt/layers/block.py             (+2 lines)  - 参数接口
vggt/models/vggt.py              (+4 lines)  - 模型配置
vggt/models/aggregator.py        (+6 lines)  - 聚合器配置
eval/eval_scannet.py             (+12 lines) - 命令行接口
tests/compare_norm_guided.py     (新文件)     - 结果对比分析脚本
tests/monitor_tests.sh           (新文件)     - 实时进度监控脚本
```

#### 3. Git提交记录
```
fe90aae - Add use_norm_guided and merge_threshold parameters to Block class
604c758 - Fix all indentation errors in merge.py grid-based split logic
2f1a46b - Fix indentation error in merge.py
33fba8f - Implement Norm-Guided Anchoring & Threshold-Gated Anti-Collapse optimizations
```

### 🔄 正在进行的测试

#### Baseline Test (网格划分方法)
```bash
python eval/eval_scannet.py \
    --merging 0 --merge_ratio 0.9 \
    --input_frame 100 --num_scenes 5 \
    --output_path ./tests/tests_result/baseline_grid_5scenes
```
- **状态**: 运行中 (Terminal ID: 2a511954-9e99-48fc-8f76-ef75a9135791)
- **进度**: ~60% (3/5 scenes or ~1.5 scenes per method)

#### Improved Test (Norm-Guided方法)
``bash
python eval/eval_scannet.py \
    --merging 0 --merge_ratio 0.9 \
    --use_norm_guided --merge_threshold 0.85 \
    --input_frame 100 --num_scenes 5 \
    --output_path ./tests/tests_result/improved_norm_guided_5scenes
```
- **状态**: 运行中 (Terminal ID: 7880c337-a5c2-40f8-9e12-7f024998f455)
- **进度**: ~60% (同步进行)

### 📋 待完成任务

1. **等待测试完成** (~10-20分钟)
   - 5个ScanNet场景，每个100帧
   - 预计总时间：~25-30分钟（两个测试并行）

2. **运行对比分析**
   ```bash
   python tests/compare_norm_guided.py
   ```
   将生成：
   - 表格对比：CD, ATE, ARE, Time等指标
   - 改进百分比计算
   - JSON格式结果保存

3. **结果解读**
   - 如果CD改善≥10%：成功验证
   - 如果CD改善0-10%：适度改进
   - 如果CD退化：需要调试参数（考虑调整分层比例或阈值）

### 🔍 关键设计决策

#### 方案一参数选择理由
- **10% protected**: 参考DINOv2的register tokens比例
- **40% dst anchors**: 提供足够的高质量锚点供匹配
- **50% src**: 平衡合并率与质量

#### 方案二参数选择理由
- **τ=0.85**: 平衡严格性和灵活性
  - τ=0.80: 更宽松，合并更多但可能引入噪声
  - τ=0.90: 更严格，保留更多但可能速度变慢

### 📊 预期结果

#### 成功标准
1. ✅ **Chamfer Distance** ↓ 10-30% (3D重建质量)
2. ✅ **ATE/ARE** 保持或改善 (位姿精度)
3. ✅ **Inference Time** ±10% (速度开销可控)

#### 理论依据
- **Vision Transformers Need Registers** (ICCV 2023): 高norm tokens对应高信息量
- **DINOv2 论文**: Register tokens保护机制
- **DynamicViT**: 自适应token pruning

### 🛠 如何使用新功能

#### 使用Norm-Guided Anchoring
```bash
python eval/eval_scannet.py --use_norm_guided ...
```

#### 调整相似度阈值
```bash
python eval/eval_scannet.py --merge_threshold 0.90 ...
```

#### 同时启用两种优化
```bash
python eval/eval_scannet.py --use_norm_guided --merge_threshold 0.85 ...
```

#### 回退到Grid-Based方法
```bash
python eval/eval_scannet.py  # 不添加--use_norm_guided即可
```

### 📝 后续工作

1. **参数扫描实验** (如果时间充足)
   - 测试不同分层比例: 5-35-60, 15-35-50
   - 测试不同阈值: 0.80, 0.85, 0.90

2. **可视化分析** (可选)
   - 绘制被保护的高norm tokens在图像中的分布
   - 可视化被拒绝合并的token pairs

3. **性能优化** (如果出现速度退化)
   - Profile norm计算和argsort操作
   - 考虑使用topk代替full sort

---

**当前时间**: 2026-02-27 15:30  
**分支**: norm-guided-merge  
**最新commit**: fe90aae
