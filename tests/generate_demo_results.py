#!/usr/bin/env python3
"""
Generate illustrative demo data for hybrid registration results.
NOTICE: ILLUSTRATIVE DEMO DATA - NOT REAL EXPERIMENTAL RESULTS.

Design goals:
1) 普通场景：误差小，达到 AR 可用状态
2) 大基线场景：误差增大，但仍处于勉强可用范围
3) 方法关系：C 整体优于 B，允许少量次优样本
"""

from pathlib import Path
import numpy as np
import pandas as pd


np.random.seed(42)

workspace = Path('/home/hba/Documents/FastVGGT')
input_pair_csv = workspace / 'tests/tests_result/hybrid_registration_7scenes/hybrid_registration_pair_results.csv'
output_dir = workspace / 'tests/tests_result/hybrid_registration_7scenes/demo'
output_dir.mkdir(parents=True, exist_ok=True)

output_pair_csv = output_dir / 'hybrid_registration_pair_results_demo.csv'
output_summary_csv = output_dir / 'hybrid_registration_summary_demo.csv'

print(f"[DEMO] Loading base pair data: {input_pair_csv}")
df_base = pd.read_csv(input_pair_csv)

pair_keys = ['scene', 'sequence', 'src_frame', 'dst_frame']

# 每个 pair 分配一个场景级别：普通 / 大基线
pair_meta = (
    df_base[pair_keys + ['baseline_rot_deg', 'baseline_trans_m']]
    .drop_duplicates(pair_keys)
    .copy()
)

# 用基线强度排序后分层：低强度 -> 普通场景；高强度 -> 大基线
pair_meta['baseline_score'] = pair_meta['baseline_rot_deg'] + 35.0 * pair_meta['baseline_trans_m']
q = pair_meta['baseline_score'].quantile(0.45)
pair_meta['regime'] = np.where(pair_meta['baseline_score'] <= q, '普通场景', '大基线场景')

# 为了让 demo 更接近叙述，对普通场景的 baseline 重新缩放到低范围
ordinary_mask = pair_meta['regime'] == '普通场景'
large_mask = pair_meta['regime'] == '大基线场景'

pair_meta.loc[ordinary_mask, 'baseline_rot_demo'] = np.clip(
    np.random.normal(7.0, 2.0, ordinary_mask.sum()), 2.0, 12.0
)
pair_meta.loc[ordinary_mask, 'baseline_trans_demo'] = np.clip(
    np.random.normal(0.14, 0.04, ordinary_mask.sum()), 0.05, 0.24
)

pair_meta.loc[large_mask, 'baseline_rot_demo'] = np.clip(
    np.random.normal(31.0, 8.0, large_mask.sum()), 18.0, 55.0
)
pair_meta.loc[large_mask, 'baseline_trans_demo'] = np.clip(
    np.random.normal(0.62, 0.22, large_mask.sum()), 0.30, 1.20
)

meta_cols = pair_keys + ['regime', 'baseline_rot_demo', 'baseline_trans_demo']
df_demo = df_base.copy().merge(pair_meta[meta_cols], on=pair_keys, how='left')

# 目标误差分布（示意）：
# 普通场景：AR 可用；大基线：勉强可用
# 注意：这里是 demo 叙述数据，不代表真实实验值
profile = {
    '普通场景': {
        'A': {'rot': (3.4, 0.7), 'trans': (0.066, 0.012), 'cd': (0.064, 0.010), 'rt': (1080, 80), 'it': (28, 6)},
        'B': {'rot': (2.2, 0.5), 'trans': (0.043, 0.009), 'cd': (0.048, 0.008), 'rt': (690, 45), 'it': (0, 0)},
        'C': {'rot': (1.7, 0.4), 'trans': (0.034, 0.007), 'cd': (0.041, 0.007), 'rt': (1410, 85), 'it': (20, 5)},
    },
    '大基线场景': {
        'A': {'rot': (11.8, 2.2), 'trans': (0.320, 0.060), 'cd': (0.165, 0.022), 'rt': (1260, 95), 'it': (44, 8)},
        'B': {'rot': (7.6, 1.6), 'trans': (0.205, 0.040), 'cd': (0.119, 0.018), 'rt': (725, 55), 'it': (0, 0)},
        'C': {'rot': (6.2, 1.3), 'trans': (0.165, 0.032), 'cd': (0.104, 0.015), 'rt': (1660, 110), 'it': (33, 7)},
    },
}

# 生成每个 pair 的 A/B/C 误差，保持同一 pair 的统计相关性
for pair_value, group_idx in df_demo.groupby(pair_keys).groups.items():
    idxs = list(group_idx)
    regime = df_demo.loc[idxs[0], 'regime']

    # pair-level difficulty jitter
    pair_jitter = np.clip(np.random.normal(1.0, 0.09), 0.82, 1.22)

    method_rows = {m: None for m in ['A', 'B', 'C']}
    for idx in idxs:
        method = df_demo.loc[idx, 'method']
        if method in method_rows:
            method_rows[method] = idx

    for method, idx in method_rows.items():
        if idx is None:
            continue
        p = profile[regime][method]

        rot = np.random.normal(p['rot'][0], p['rot'][1]) * pair_jitter
        trans = np.random.normal(p['trans'][0], p['trans'][1]) * pair_jitter
        cd = np.random.normal(p['cd'][0], p['cd'][1]) * pair_jitter
        rt = np.random.normal(p['rt'][0], p['rt'][1])

        if method == 'B':
            iters = -1
        else:
            iters = int(np.clip(np.random.normal(p['it'][0], p['it'][1]), 8, 70))

        df_demo.loc[idx, 'rotation_error_deg'] = rot
        df_demo.loc[idx, 'translation_error_m'] = trans
        df_demo.loc[idx, 'chamfer_distance'] = cd
        df_demo.loc[idx, 'runtime_ms'] = rt
        df_demo.loc[idx, 'icp_iterations'] = iters

    # 允许少量“C 次优”样本（更像真实波动）
    c_idx = method_rows['C']
    b_idx = method_rows['B']
    if c_idx is not None and b_idx is not None:
        if regime == '普通场景' and np.random.rand() < 0.10:
            metric = np.random.choice(['rotation_error_deg', 'translation_error_m', 'chamfer_distance'])
            df_demo.loc[c_idx, metric] = df_demo.loc[b_idx, metric] * np.random.uniform(1.02, 1.08)
        if regime == '大基线场景' and np.random.rand() < 0.20:
            metric = np.random.choice(['rotation_error_deg', 'translation_error_m', 'chamfer_distance'])
            df_demo.loc[c_idx, metric] = df_demo.loc[b_idx, metric] * np.random.uniform(1.03, 1.10)

# Clip to keep within target ranges
# 普通场景（AR 可用）
mask_o = df_demo['regime'] == '普通场景'
df_demo.loc[mask_o, 'rotation_error_deg'] = df_demo.loc[mask_o, 'rotation_error_deg'].clip(0.7, 5.0)
df_demo.loc[mask_o, 'translation_error_m'] = df_demo.loc[mask_o, 'translation_error_m'].clip(0.010, 0.090)
df_demo.loc[mask_o, 'chamfer_distance'] = df_demo.loc[mask_o, 'chamfer_distance'].clip(0.018, 0.100)

# 大基线场景（勉强可用）
mask_l = df_demo['regime'] == '大基线场景'
df_demo.loc[mask_l, 'rotation_error_deg'] = df_demo.loc[mask_l, 'rotation_error_deg'].clip(3.0, 14.0)
df_demo.loc[mask_l, 'translation_error_m'] = df_demo.loc[mask_l, 'translation_error_m'].clip(0.080, 0.420)
df_demo.loc[mask_l, 'chamfer_distance'] = df_demo.loc[mask_l, 'chamfer_distance'].clip(0.060, 0.230)

# 回填 baseline（demo）
df_demo['baseline_rot_deg'] = df_demo['baseline_rot_demo']
df_demo['baseline_trans_m'] = df_demo['baseline_trans_demo']
df_demo.drop(columns=['baseline_rot_demo', 'baseline_trans_demo'], inplace=True)

# 按原脚本 recall 标准计算（5° + 0.05m）
df_demo['recall_hit'] = (
    (df_demo['rotation_error_deg'] < 5.0) & (df_demo['translation_error_m'] < 0.05)
).astype(int)

# 标记数据性质
if 'data_type' in df_demo.columns:
    df_demo['data_type'] = '[DEMO]'
else:
    df_demo.insert(0, 'data_type', '[DEMO]')

# 保存 pair 结果
df_demo.to_csv(output_pair_csv, index=False)
print(f"[DEMO] Saved pair results: {output_pair_csv}")


# ===== 汇总：按 regime + scene + method =====
agg_cols = {
    'rotation_error_deg': 'mean',
    'translation_error_m': 'mean',
    'chamfer_distance': 'mean',
    'icp_iterations': lambda x: float(np.mean([v for v in x if int(v) > 0])) if np.any(np.array(x) > 0) else 0.0,
    'runtime_ms': 'mean',
    'recall_hit': 'mean',
}

summary_rows = []

for regime in ['普通场景', '大基线场景']:
    sub_r = df_demo[df_demo['regime'] == regime]
    for scene in sorted(sub_r['scene'].unique()):
        for method in ['A', 'B', 'C']:
            g = sub_r[(sub_r['scene'] == scene) & (sub_r['method'] == method)]
            if g.empty:
                continue
            summary_rows.append({
                'regime': regime,
                'scene': scene,
                'method': method,
                'num_pairs': int(len(g)),
                'recall': float(g['recall_hit'].mean()),
                'mean_rotation_error_deg': float(g['rotation_error_deg'].mean()),
                'mean_translation_error_m': float(g['translation_error_m'].mean()),
                'mean_chamfer_distance': float(g['chamfer_distance'].mean()),
                'mean_icp_iterations': float(np.mean([v for v in g['icp_iterations'].tolist() if int(v) > 0])) if method != 'B' else 0.0,
                'mean_runtime_ms': float(g['runtime_ms'].mean()),
            })

    # regime-ALL
    for method in ['A', 'B', 'C']:
        g = sub_r[sub_r['method'] == method]
        if g.empty:
            continue
        summary_rows.append({
            'regime': regime,
            'scene': 'ALL',
            'method': method,
            'num_pairs': int(len(g)),
            'recall': float(g['recall_hit'].mean()),
            'mean_rotation_error_deg': float(g['rotation_error_deg'].mean()),
            'mean_translation_error_m': float(g['translation_error_m'].mean()),
            'mean_chamfer_distance': float(g['chamfer_distance'].mean()),
            'mean_icp_iterations': float(np.mean([v for v in g['icp_iterations'].tolist() if int(v) > 0])) if method != 'B' else 0.0,
            'mean_runtime_ms': float(g['runtime_ms'].mean()),
        })

# 总体 ALL（跨 regime）
for method in ['A', 'B', 'C']:
    g = df_demo[df_demo['method'] == method]
    summary_rows.append({
        'regime': '总体',
        'scene': 'ALL',
        'method': method,
        'num_pairs': int(len(g)),
        'recall': float(g['recall_hit'].mean()),
        'mean_rotation_error_deg': float(g['rotation_error_deg'].mean()),
        'mean_translation_error_m': float(g['translation_error_m'].mean()),
        'mean_chamfer_distance': float(g['chamfer_distance'].mean()),
        'mean_icp_iterations': float(np.mean([v for v in g['icp_iterations'].tolist() if int(v) > 0])) if method != 'B' else 0.0,
        'mean_runtime_ms': float(g['runtime_ms'].mean()),
    })

df_summary = pd.DataFrame(summary_rows)

# 排序，便于阅读
regime_order = {'普通场景': 0, '大基线场景': 1, '总体': 2}
method_order = {'A': 0, 'B': 1, 'C': 2}
scene_order = {'office': 0, 'redkitchen': 1, 'ALL': 2}
df_summary['__r'] = df_summary['regime'].map(regime_order)
df_summary['__s'] = df_summary['scene'].map(scene_order).fillna(9)
df_summary['__m'] = df_summary['method'].map(method_order)
df_summary = df_summary.sort_values(['__r', '__s', '__m']).drop(columns=['__r', '__s', '__m'])

df_summary.to_csv(output_summary_csv, index=False)
print(f"[DEMO] Saved summary: {output_summary_csv}")

print('\n[DEMO] ===== Quick Check: ALL (普通场景) =====')
normal_all = df_summary[(df_summary['regime'] == '普通场景') & (df_summary['scene'] == 'ALL')]
print(normal_all[['method', 'mean_rotation_error_deg', 'mean_translation_error_m', 'mean_chamfer_distance']].to_string(index=False))

print('\n[DEMO] ===== Quick Check: ALL (大基线场景) =====')
large_all = df_summary[(df_summary['regime'] == '大基线场景') & (df_summary['scene'] == 'ALL')]
print(large_all[['method', 'mean_rotation_error_deg', 'mean_translation_error_m', 'mean_chamfer_distance']].to_string(index=False))

print('\n[DEMO] IMPORTANT: This is ILLUSTRATIVE DEMO DATA ONLY.')
