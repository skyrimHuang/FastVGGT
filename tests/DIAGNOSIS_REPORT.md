# Norm-Guided Implementation Results & Analysis

## 测试结果总结

**测试配置**:
- 数据集: ScanNet (5个场景)
- 每场景帧数: 100
- Merge ratio: 0.9
- Threshold: 0.85
- 分层配置: 10% protected, 40% dst, 50% src

## 结果对比

| Metric | Baseline | Improved | Change | Status |
|--------|----------|----------|---------|--------|
| **Chamfer Distance** | 0.5229 | 0.5958 | **+13.95%** | ✗ 退化 |
| **ATE** | 0.541 | 2.127 | **+293%** | ✗ 退化 |
| **ARE** | 12.16° | 90.63° | **+645%** | ✗ 退化 |
| **RPE Rot** | 15.79° | 41.48° | **+163%** | ✗ 退化 |
| **RPE Trans** | 0.419 | 0.977 | **+134%** | ✗ 退化 |
| **Inference Time** | 19162ms | 22911ms | **+19.6%** | ✗ 变慢 |

**结论**: 所有指标均显著退化，需要深入调查根本原因。

---

## 问题诊断

### 🔴 Critical Issue 1: 位姿精度崩溃 (ATE↑293%, ARE↑645%)

**现象**: 相机位姿估计几乎完全失效

**可能原因**:
1. **保护了错误的tokens**: 并非所有高norm tokens都对位姿估计有用
   - 高norm可能来自噪声、异常值、光照变化
   - 保护这些tokens反而引入错误信息

2. **破坏了第一帧参考**: 
   - 原始代码将第一帧所有tokens标记为dst（Lines 136-137）
   - Norm-guided可能打乱了这个机制
   - 检查代码：是否在use_norm_guided=True时仍保留第一帧特殊处理？

3. **跨帧token匹配被破坏**:
   - 多视图几何需要跨帧一致的feature correspondence
   - Norm-based划分可能导致跨帧dst/src不一致

### 🔴 Critical Issue 2: 3D重建质量下降 (CD↑13.95%)

**现象**: 点云重建误差增加

**可能原因**:
1. **低norm tokens包含重要几何信息**: 
   - 平坦表面（墙面）虽然norm低，但对3D结构重要
   - 过度保护高norm可能丢失整体结构

2. **合并率实际降低**:
   - Threshold=0.85可能过于严格
   - 检查实际合并了多少tokens（r_actual vs r）

### 🔴 Critical Issue 3: 速度退化 (Time↑19.6%)

**现象**: 推理时间增加~3750ms

**可能原因**:
1. **Norm计算+排序开销**: 理论<72ms，实际可能更大
2. **Threshold过滤减少合并**: 保留更多tokens → 后续层计算量增加
3. **GPU同步点**: argsort可能引入CPU-GPU同步

---

## 根本原因分析

### 推测：第一帧处理逻辑缺失

查看代码 `merge.py` Lines 90-127:

```python
if use_norm_guided and enable_protection:
    # Norm-based split for ALL tokens
    token_norms = metric.norm(dim=-1)
    sorted_indices = torch.argsort(avg_norms, descending=True)
    ...
```

**问题**: 
- 没有保留原始代码中"第一帧全部作为dst"的逻辑！
- 原始代码Lines 136-137:
  ```python
  if num_imgs > 0:
      idx_buffer_seq[:tokens_per_img] = -1  # 第一帧全dst
  ```
- Norm-guided分支完全跳过了这个逻辑

**影响**: 
- 第一帧的tokens可能被分到src，导致参考帧丢失
- 多视图几何的anchor frame被破坏
- 位姿估计失去全局一致性

---

## 修复方案

### 方案A: 保留第一帧特殊处理 (推荐)

修改 `merge.py` Lines 90-127:

```python
if use_norm_guided and enable_protection:
    # Compute norm
    token_norms = metric.norm(dim=-1)
    avg_norms = token_norms.mean(dim=0)
    
    # 🔧 FIX: Mark first frame tokens as dst BEFORE norm-based split
    idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
    idx_buffer_seq[:tokens_per_img] = -1  # First frame all dst
    
    # Apply norm-based split only to remaining frames
    remaining_indices = torch.arange(tokens_per_img, N, device=metric.device)
    remaining_norms = avg_norms[tokens_per_img:]
    sorted_remaining = torch.argsort(remaining_norms, descending=True)
    
    # Offset indices back to global
    sorted_indices = sorted_remaining + tokens_per_img
    
    # Continue with 3-tier split on remaining tokens...
```

### 方案B: 降低Threshold (快速测试)

```bash
# 测试更宽松的阈值
python eval/eval_scannet.py --use_norm_guided --merge_threshold 0.70 ...
```

### 方案C: 调整分层比例

```python
# 更保守的保护策略
num_protected = int(N * 0.05)  # 5% instead of 10%
num_dst = int(N * 0.45)        # 45% instead of 40%
```

### 方案D: 禁用Threshold，单独测试Norm-Guided

```bash
python eval/eval_scannet.py --use_norm_guided --merge_threshold 0.0 ...
```

---

## 后续实验建议

### 立即执行 (验证假设)

1. **Test 1: 修复第一帧逻辑**
   - 实施方案A
   - 预期: ATE/ARE显著改善

2. **Test 2: 单独测试Norm-Guided**
   - `--use_norm_guided --merge_threshold 0.0`
   - 验证norm-guided本身的影响

3. **Test 3: 单独测试Threshold**
   - 不加`--use_norm_guided`，只设`--merge_threshold 0.85`
   - 验证threshold的影响

### 详细分析 (如果时间充足)

4. **添加调试输出**:
   ```python
   # In merge.py after split
   print(f"[DEBUG] First frame in dst: {(dst_indices < tokens_per_img).sum().item()}")
   print(f"[DEBUG] Actual merge count: {r_actual} / {r}")
   ```

5. **可视化分析**:
   - 绘制被保护tokens的norm分布
   - 绘制第一帧vs其他帧的dst/src分布
   - 可视化被threshold拒绝的token pairs

---

## 经验教训

### ✅ 成功的方面
- 代码架构清晰，参数传递链路完整
- 测试框架工作正常
- 对比分析脚本有效

### ❌ 失败的方面
- **未充分理解原始算法的关键约束**（第一帧作为全局参考）
- **未进行消融实验**（应先单独测试每个改进）
- **未添加调试输出**（无法快速定位问题）

### 📚 学到的启示
1. **在修改复杂系统前，必须完全理解原始设计的rationale**
2. **Always do ablation studies**: 单独测试每个改进，再组合
3. **Add instrumentation early**: 调试输出应该和代码同时写
4. **理论听起来合理 ≠ 实际会work**: 需要实验验证

---

## 下一步行动

### Immediate (30分钟内)
1. 修复第一帧处理逻辑（方案A）
2. 重新运行测试（1个场景快速验证）

### Short-term (1-2小时)
3. 如果修复后仍有问题，进行消融实验（Tests 2-3）
4. 添加详细调试输出

### Medium-term (如果继续优化)
5. 尝试其他norm-guided策略（例如：只在非第一帧上应用）
6. 参数扫描（不同分层比例和阈值）
7. 可视化分析

---

## 文件清单

### 结果文件
- `tests/tests_result/baseline_grid_5scenes/` - Baseline结果
- `tests/tests_result/improved_norm_guided_5scenes/` - Improved结果
- `tests/tests_result/comparison_norm_guided.json` - 对比数据

### 代码文件
- `merging/merge.py` - 核心实现（需要修复）
- `vggt/layers/attention.py` - 参数传递
- `vggt/layers/block.py` - Block接口
- `eval/eval_scannet.py` - 测试入口

### 文档文件
- `tests/NORM_GUIDED_README.md` - 完整文档
- `tests/IMPLEMENTATION_STATUS.md` - 实施状态
- `tests/DIAGNOSIS_REPORT.md` - 本文件（问题诊断）

---

**创建时间**: 2026-02-27 15:40  
**分支**: norm-guided-merge  
**状态**: ⚠️ Implementation complete, but results show regression - needs debugging
