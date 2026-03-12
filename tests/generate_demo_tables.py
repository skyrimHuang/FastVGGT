#!/usr/bin/env python3
"""
Generate LaTeX/CSV tables for demo hybrid registration results.
NOTICE: ILLUSTRATIVE DEMO DATA - NOT REAL EXPERIMENTAL RESULTS.

Table order follows thesis narrative:
1) 普通场景性能（AR 可用）
2) 大基线场景性能（误差增大但勉强可用）
"""

from pathlib import Path
import pandas as pd

workspace = Path('/home/hba/Documents/FastVGGT')
summary_csv = workspace / 'tests/tests_result/hybrid_registration_7scenes/demo/hybrid_registration_summary_demo.csv'
output_dir = workspace / 'tests/tests_result/hybrid_registration_7scenes/demo'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[TABLE] Loading summary: {summary_csv}")
df = pd.read_csv(summary_csv)


def get_regime_all(regime: str) -> pd.DataFrame:
    out = df[(df['regime'] == regime) & (df['scene'] == 'ALL')].copy()
    order = {'A': 0, 'B': 1, 'C': 2}
    out['__o'] = out['method'].map(order)
    out = out.sort_values('__o').drop(columns='__o')
    return out


normal_all = get_regime_all('普通场景')
large_all = get_regime_all('大基线场景')

if normal_all.empty or large_all.empty:
    raise RuntimeError('[TABLE] Missing regime summary rows. Run generate_demo_results.py first.')


# ========= Table 1: 普通场景 =========
print('[TABLE] Generating Table 1 (普通场景)...')

latex_table1 = r"""
\begin{table}[h]
\centering
\caption{普通场景下三种方法的整体性能对比}
\label{table:normal_scene_performance}
\begin{tabular}{ccccc}
\toprule
方法 & 旋转误差 (度) & 平移误差 (米) & Chamfer距离 & 样本数 \\
\midrule
"""

for _, row in normal_all.iterrows():
    latex_table1 += (
        f"方法 {row['method']} & "
        f"{row['mean_rotation_error_deg']:.2f} & "
        f"{row['mean_translation_error_m']:.4f} & "
        f"{row['mean_chamfer_distance']:.4f} & "
        f"{int(row['num_pairs'])} \\\n"
    )

b = normal_all[normal_all['method'] == 'B'].iloc[0]
c = normal_all[normal_all['method'] == 'C'].iloc[0]
rot_imp_n = (1 - c['mean_rotation_error_deg'] / b['mean_rotation_error_deg']) * 100
trans_imp_n = (1 - c['mean_translation_error_m'] / b['mean_translation_error_m']) * 100
cd_imp_n = (1 - c['mean_chamfer_distance'] / b['mean_chamfer_distance']) * 100

latex_table1 += (
    "\\midrule\n"
    f"C相对B改进 & {rot_imp_n:.2f}\\% & {trans_imp_n:.2f}\\% & {cd_imp_n:.2f}\\% & - \\\n"
)

latex_table1 += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small{\textit{注：} 演示数据，仅用于排版。普通场景误差设定为 AR 可用量级。}
\end{flushleft}
\end{table}
"""

(output_dir / 'demo_table1_scene_comparison.tex').write_text(latex_table1, encoding='utf-8')


# ========= Table 2: 大基线场景 =========
print('[TABLE] Generating Table 2 (大基线场景)...')

latex_table2 = r"""
\begin{table}[h]
\centering
\caption{大基线场景下三种方法的整体性能对比}
\label{table:large_baseline_performance}
\begin{tabular}{ccccc}
\toprule
方法 & 旋转误差 (度) & 平移误差 (米) & Chamfer距离 & 样本数 \\
\midrule
"""

for _, row in large_all.iterrows():
    latex_table2 += (
        f"方法 {row['method']} & "
        f"{row['mean_rotation_error_deg']:.2f} & "
        f"{row['mean_translation_error_m']:.4f} & "
        f"{row['mean_chamfer_distance']:.4f} & "
        f"{int(row['num_pairs'])} \\\n"
    )

b = large_all[large_all['method'] == 'B'].iloc[0]
c = large_all[large_all['method'] == 'C'].iloc[0]
rot_imp_l = (1 - c['mean_rotation_error_deg'] / b['mean_rotation_error_deg']) * 100
trans_imp_l = (1 - c['mean_translation_error_m'] / b['mean_translation_error_m']) * 100
cd_imp_l = (1 - c['mean_chamfer_distance'] / b['mean_chamfer_distance']) * 100

latex_table2 += (
    "\\midrule\n"
    f"C相对B改进 & {rot_imp_l:.2f}\\% & {trans_imp_l:.2f}\\% & {cd_imp_l:.2f}\\% & - \\\n"
)

latex_table2 += r"""\bottomrule
\end{tabular}
\begin{flushleft}
\small{\textit{注：} 演示数据，仅用于排版。大基线场景误差增大，但仍保持在勉强可用范围。}
\end{flushleft}
\end{table}
"""

(output_dir / 'demo_table2_overall_performance.tex').write_text(latex_table2, encoding='utf-8')


# ========= CSV summary table =========
rows = []
for regime_name, sub in [('普通场景', normal_all), ('大基线场景', large_all)]:
    for _, r in sub.iterrows():
        rows.append({
            'Regime': regime_name,
            'Method': r['method'],
            'Rotation Error (deg)': round(float(r['mean_rotation_error_deg']), 3),
            'Translation Error (m)': round(float(r['mean_translation_error_m']), 4),
            'Chamfer Distance': round(float(r['mean_chamfer_distance']), 4),
            'Samples': int(r['num_pairs']),
        })

df_csv = pd.DataFrame(rows)
df_csv.to_csv(output_dir / 'demo_table_markdown.csv', index=False, encoding='utf-8')

print('[TABLE] Saved:')
print(f"  - {output_dir / 'demo_table1_scene_comparison.tex'}")
print(f"  - {output_dir / 'demo_table2_overall_performance.tex'}")
print(f"  - {output_dir / 'demo_table_markdown.csv'}")
print('[TABLE] IMPORTANT: ILLUSTRATIVE DEMO DATA ONLY')
