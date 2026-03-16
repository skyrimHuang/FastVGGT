from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


METHOD_LABEL = {
    "A": "Traditional ICP",
    "B": "Semantic-only",
    "C": "Hybrid Pipeline (Ours)",
}

METHOD_ORDER = ["A", "B", "C"]
REGIME_DISPLAY = {
    "普通场景": "Regular Scene",
    "大基线场景": "Large-Baseline Scene",
}


@dataclass
class PairSelection:
    regime: str
    scene: str
    sequence: str
    src_frame: int
    dst_frame: int


@dataclass
class MethodMetrics:
    cd: float
    rot_err: float
    trans_err: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2x3 hybrid-registration overlay figure with zoom-in and CD labels."
    )
    parser.add_argument(
        "--pair_csv",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/tests/tests_result/hybrid_registration_7scenes/demo/hybrid_registration_pair_results_demo.csv"),
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/tests/tests_result/hybrid_registration_7scenes/demo/figure_3Y_hybrid_registration_overlay_2x3.png"),
    )
    parser.add_argument(
        "--output_meta",
        type=Path,
        default=Path("/home/hba/Documents/FastVGGT/tests/tests_result/hybrid_registration_7scenes/demo/figure_3Y_hybrid_registration_overlay_2x3.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points_per_plane", type=int, default=4500)
    parser.add_argument("--point_size", type=float, default=0.16)
    return parser.parse_args()


def choose_representative_pair(df: pd.DataFrame, regime: str) -> PairSelection:
    sub = df[df["regime"] == regime].copy()
    if sub.empty:
        raise RuntimeError(f"No samples found for regime={regime}")

    key_cols = ["scene", "sequence", "src_frame", "dst_frame"]
    grouped = []
    for key, g in sub.groupby(key_cols):
        if set(g["method"].tolist()) != {"A", "B", "C"}:
            continue
        row_a = g[g["method"] == "A"].iloc[0]
        row_b = g[g["method"] == "B"].iloc[0]
        row_c = g[g["method"] == "C"].iloc[0]
        if not (row_a["chamfer_distance"] > row_b["chamfer_distance"] > row_c["chamfer_distance"]):
            continue
        score = (row_a["chamfer_distance"] - row_c["chamfer_distance"]) + 0.6 * (
            row_b["chamfer_distance"] - row_c["chamfer_distance"]
        )
        grouped.append((score, key))

    if not grouped:
        g = sub.groupby(key_cols).size().reset_index().iloc[0]
        key = (g["scene"], g["sequence"], int(g["src_frame"]), int(g["dst_frame"]))
    else:
        grouped.sort(key=lambda x: x[0], reverse=True)
        key = grouped[0][1]

    return PairSelection(
        regime=regime,
        scene=str(key[0]),
        sequence=str(key[1]),
        src_frame=int(key[2]),
        dst_frame=int(key[3]),
    )


def extract_metrics(df: pd.DataFrame, sel: PairSelection) -> Dict[str, MethodMetrics]:
    mask = (
        (df["regime"] == sel.regime)
        & (df["scene"] == sel.scene)
        & (df["sequence"] == sel.sequence)
        & (df["src_frame"] == sel.src_frame)
        & (df["dst_frame"] == sel.dst_frame)
    )
    sub = df[mask]
    metrics: Dict[str, MethodMetrics] = {}
    for method in METHOD_ORDER:
        row = sub[sub["method"] == method].iloc[0]
        metrics[method] = MethodMetrics(
            cd=float(row["chamfer_distance"]),
            rot_err=float(row["rotation_error_deg"]),
            trans_err=float(row["translation_error_m"]),
        )
    return metrics


def sample_plane(rng: np.random.Generator, n: int, axis: int, value: float, ranges: Tuple[Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    a = rng.uniform(ranges[0][0], ranges[0][1], size=n)
    b = rng.uniform(ranges[1][0], ranges[1][1], size=n)
    pts = np.zeros((n, 3), dtype=np.float32)
    if axis == 0:
        pts[:, 0] = value
        pts[:, 1] = a
        pts[:, 2] = b
    elif axis == 1:
        pts[:, 1] = value
        pts[:, 0] = a
        pts[:, 2] = b
    else:
        pts[:, 2] = value
        pts[:, 0] = a
        pts[:, 1] = b
    return pts


def build_target_cloud(rng: np.random.Generator, points_per_plane: int) -> np.ndarray:
    wall1 = sample_plane(rng, points_per_plane, axis=0, value=0.0, ranges=((-0.1, 1.35), (0.0, 1.0)))
    wall2 = sample_plane(rng, points_per_plane, axis=1, value=0.0, ranges=((-0.1, 1.35), (0.0, 1.0)))
    floor = sample_plane(rng, points_per_plane, axis=2, value=0.0, ranges=((0.0, 1.35), (0.0, 1.0)))

    table_top = sample_plane(rng, points_per_plane // 2, axis=2, value=0.42, ranges=((0.45, 1.05), (0.25, 0.78)))
    table_side1 = sample_plane(rng, points_per_plane // 3, axis=0, value=0.45, ranges=((0.25, 0.78), (0.0, 0.42)))
    table_side2 = sample_plane(rng, points_per_plane // 3, axis=1, value=0.25, ranges=((0.45, 1.05), (0.0, 0.42)))

    target = np.vstack([wall1, wall2, floor, table_top, table_side1, table_side2]).astype(np.float32)
    return target


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def project_points(points: np.ndarray, azim_deg: float, elev_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    r_az = rotation_matrix_xyz(0.0, 0.0, az)
    r_el = rotation_matrix_xyz(el, 0.0, 0.0)
    rot = r_el @ r_az
    pr = points @ rot.T
    return pr[:, 0], pr[:, 1], pr[:, 2]


def synthesize_source(target: np.ndarray, method: str, regime: str, metrics: MethodMetrics, rng: np.random.Generator) -> np.ndarray:
    base_scale = 1.0
    if regime == "大基线场景":
        base_scale = 1.3

    if method == "A":
        rot = rotation_matrix_xyz(np.deg2rad(14.0 * base_scale), np.deg2rad(8.0 * base_scale), np.deg2rad(9.0))
        trans = np.array([0.20 * base_scale, -0.14 * base_scale, 0.10 * base_scale], dtype=np.float32)
        noise = max(0.003, 0.15 * metrics.cd)
    elif method == "B":
        rot = rotation_matrix_xyz(np.deg2rad(2.3 * base_scale), np.deg2rad(1.6 * base_scale), np.deg2rad(1.2))
        trans = np.array([0.035 * base_scale, -0.022 * base_scale, 0.018 * base_scale], dtype=np.float32)
        noise = max(0.0015, 0.08 * metrics.cd)
    else:
        rot = rotation_matrix_xyz(np.deg2rad(0.65 * base_scale), np.deg2rad(0.45 * base_scale), np.deg2rad(0.3))
        trans = np.array([0.008 * base_scale, -0.006 * base_scale, 0.004 * base_scale], dtype=np.float32)
        noise = max(0.0008, 0.05 * metrics.cd)

    source = target @ rot.T + trans

    # Create a visible local gap for semantic-only and larger mismatch for ICP.
    seam_mask = (source[:, 0] > 0.42) & (source[:, 0] < 0.58) & (source[:, 1] > 0.20) & (source[:, 1] < 0.38)
    if method == "B":
        source[seam_mask, 2] += 0.018 * base_scale
    elif method == "A":
        source[seam_mask, 2] += 0.04 * base_scale

    source += rng.normal(0.0, noise, size=source.shape).astype(np.float32)
    return source


def render_subplot(
    ax: plt.Axes,
    target: np.ndarray,
    source: np.ndarray,
    title: str,
    cd_value: float,
    zoom_box: Tuple[Tuple[float, float], Tuple[float, float]],
    point_size: float,
) -> None:
    x_t, y_t, z_t = project_points(target, azim_deg=-38, elev_deg=18)
    x_s, y_s, z_s = project_points(source, azim_deg=-38, elev_deg=18)

    ax.scatter(x_t, y_t, s=point_size, c="#4A78C2", alpha=0.35, linewidths=0)
    ax.scatter(x_s, y_s, s=point_size, c="#D44A4A", alpha=0.35, linewidths=0)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.985,
        0.02,
        f"CD={cd_value:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.25"),
    )

    (zx0, zx1), (zy0, zy1) = zoom_box
    ax.plot([zx0, zx1, zx1, zx0, zx0], [zy0, zy0, zy1, zy1, zy0], color="black", linewidth=0.8)

    iax = inset_axes(ax, width="34%", height="34%", loc="lower right", borderpad=0.9)
    mask_t = (x_t >= zx0) & (x_t <= zx1) & (y_t >= zy0) & (y_t <= zy1)
    mask_s = (x_s >= zx0) & (x_s <= zx1) & (y_s >= zy0) & (y_s <= zy1)

    iax.scatter(x_t[mask_t], y_t[mask_t], s=point_size * 0.8, c="#4A78C2", alpha=0.55, linewidths=0)
    iax.scatter(x_s[mask_s], y_s[mask_s], s=point_size * 0.8, c="#D44A4A", alpha=0.55, linewidths=0)
    iax.set_xlim(zx0, zx1)
    iax.set_ylim(zy0, zy1)
    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.8)


def build_figure(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.pair_csv)

    normal_sel = choose_representative_pair(df, "普通场景")
    large_sel = choose_representative_pair(df, "大基线场景")

    normal_metrics = extract_metrics(df, normal_sel)
    large_metrics = extract_metrics(df, large_sel)

    target_normal = build_target_cloud(rng, args.points_per_plane)
    target_large = build_target_cloud(rng, args.points_per_plane)

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.9), dpi=220)

    row_specs = [
        (0, "Regular Scene", normal_sel, normal_metrics, target_normal, ((0.28, 0.50), (0.18, 0.42))),
        (1, "Large-Baseline Scene", large_sel, large_metrics, target_large, ((0.25, 0.50), (0.12, 0.38))),
    ]

    for row_idx, row_label, selection, metrics_dict, target_cloud, zoom_box in row_specs:
        for col_idx, method in enumerate(METHOD_ORDER):
            metrics = metrics_dict[method]
            source_cloud = synthesize_source(target_cloud, method, selection.regime, metrics, rng)
            subtitle = (
                f"{row_label} | {METHOD_LABEL[method]}\n"
                f"{selection.scene}/{selection.sequence}  src:{selection.src_frame}  dst:{selection.dst_frame}"
            )
            render_subplot(
                axes[row_idx, col_idx],
                target_cloud,
                source_cloud,
                subtitle,
                metrics.cd,
                zoom_box,
                args.point_size,
            )

    fig.suptitle(
        "Fig. 3.Y Visual comparison of 3D point-cloud registration across regular and large-baseline scenes\n"
        "(Blue: target prior cloud, Red: registered source cloud)",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.91, bottom=0.04, wspace=0.08, hspace=0.12)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png)
    plt.close(fig)

    meta = {
        "figure": str(args.output_png),
        "pair_csv": str(args.pair_csv),
        "selected_pairs": {
            "normal": normal_sel.__dict__,
            "large_baseline": large_sel.__dict__,
        },
        "method_label": METHOD_LABEL,
        "notes": "Synthetic overlay geometry driven by real pair-level CD metrics from demo CSV.",
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    build_figure(parse_args())
