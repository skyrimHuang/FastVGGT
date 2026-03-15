from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_relative_metrics(group: pd.DataFrame) -> pd.DataFrame:
    base = group[group["merge_ratio"] == 0.0]
    if base.empty:
        raise ValueError("Each input_frame group must contain merge_ratio=0.0 baseline")
    base_row = base.iloc[0]

    out = group.copy()
    geom_cols = ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans"]
    for col in geom_cols + ["inference_time_ms"]:
        out[f"{col}_rel"] = out[col] / base_row[col]

    out["E_rel"] = out[[f"{c}_rel" for c in geom_cols]].mean(axis=1)
    out["speedup"] = 1.0 / out["inference_time_ms_rel"]
    return out


def compute_optimal_curve(df_rel: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for input_frame, group in df_rel.groupby("input_frame", sort=True):
        g = group.copy()
        g["score"] = alpha * g["E_rel"] + (1.0 - alpha) * g["inference_time_ms_rel"]
        best = g.loc[g["score"].idxmin()]
        rows.append(
            {
                "input_frame": int(input_frame),
                "merge_ratio_opt": float(best["merge_ratio"]),
                "score": float(best["score"]),
            }
        )
    return pd.DataFrame(rows).sort_values("input_frame")


def main() -> None:
    csv_path = Path("tests/tests_result/scannet_merge_sweep/scannet_merge_sweep_summary.csv")
    out_dir = Path("tests/tests_result/scannet_merge_sweep/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df_rel = pd.concat(
        [compute_relative_metrics(g) for _, g in df.groupby("input_frame", sort=True)],
        ignore_index=True,
    )

    alpha_main = 0.6
    opt_main = compute_optimal_curve(df_rel, alpha=alpha_main)
    opt_lo = compute_optimal_curve(df_rel, alpha=0.5)
    opt_hi = compute_optimal_curve(df_rel, alpha=0.7)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)

    ax0 = axes[0]
    frames_sorted: List[int] = sorted(df_rel["input_frame"].unique().tolist())
    cmap = plt.get_cmap("viridis")

    for idx, frame in enumerate(frames_sorted):
        g = df_rel[df_rel["input_frame"] == frame].sort_values("merge_ratio")
        color = cmap(idx / max(1, len(frames_sorted) - 1))
        ax0.plot(
            g["speedup"],
            g["E_rel"],
            marker="o",
            linewidth=1.8,
            label=f"S={frame}",
            color=color,
        )
        for _, row in g.iterrows():
            ax0.annotate(
                f"r={row['merge_ratio']:.1f}",
                (row["speedup"], row["E_rel"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                color=color,
            )

    for _, row in opt_main.iterrows():
        g = df_rel[
            (df_rel["input_frame"] == row["input_frame"])
            & (np.isclose(df_rel["merge_ratio"], row["merge_ratio_opt"]))
        ].iloc[0]
        ax0.scatter(
            [g["speedup"]],
            [g["E_rel"]],
            marker="*",
            s=180,
            color="crimson",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    ax0.set_xlabel("Speedup vs r=0 baseline (×)")
    ax0.set_ylabel("Relative geometry cost E (lower is better)")
    ax0.set_title("(a) Pareto trade-off across sequence lengths")
    ax0.legend(loc="best", fontsize=8)

    ax1 = axes[1]
    ax1.plot(
        opt_lo["input_frame"],
        opt_lo["merge_ratio_opt"],
        marker="o",
        linewidth=1.4,
        label="Optimal r (alpha=0.5)",
    )
    ax1.plot(
        opt_main["input_frame"],
        opt_main["merge_ratio_opt"],
        marker="s",
        linewidth=2.2,
        label=f"Optimal r (alpha={alpha_main:.1f})",
    )
    ax1.plot(
        opt_hi["input_frame"],
        opt_hi["merge_ratio_opt"],
        marker="^",
        linewidth=1.4,
        label="Optimal r (alpha=0.7)",
    )

    for _, row in opt_main.iterrows():
        ax1.annotate(
            f"{row['merge_ratio_opt']:.1f}",
            (row["input_frame"], row["merge_ratio_opt"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax1.set_ylim(-0.02, 1.02)
    ax1.set_yticks([0.0, 0.3, 0.6, 0.9])
    ax1.set_xlabel("Sequence length S")
    ax1.set_ylabel("Selected optimal merge ratio")
    ax1.set_title("(b) Stage-wise upward shift of optimal operating point")
    ax1.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "ScanNet: Time-accuracy trade-off and optimal merge-ratio migration",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    png_path = out_dir / "scannet_merge_sweep_optimal_operating_point_combo.png"
    pdf_path = out_dir / "scannet_merge_sweep_optimal_operating_point_combo.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # Save the computed optimal curves for traceability
    opt_csv = out_dir / "scannet_merge_sweep_optimal_curve.csv"
    merged_opt = opt_lo.rename(columns={"merge_ratio_opt": "merge_ratio_opt_alpha_0_5"}).merge(
        opt_main.rename(columns={"merge_ratio_opt": "merge_ratio_opt_alpha_0_6"}),
        on="input_frame",
    ).merge(
        opt_hi.rename(columns={"merge_ratio_opt": "merge_ratio_opt_alpha_0_7"}),
        on="input_frame",
    )
    merged_opt.to_csv(opt_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {opt_csv}")


if __name__ == "__main__":
    main()
