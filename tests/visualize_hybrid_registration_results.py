#!/usr/bin/env python3
"""Visualize hybrid registration robustness results for thesis figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent / "tests_result" / "hybrid_registration_7scenes"
SUMMARY_CSV = ROOT / "hybrid_registration_summary.csv"
PAIR_CSV = ROOT / "hybrid_registration_pair_results.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_CSV)
    pairs = pd.read_csv(PAIR_CSV)
    return summary, pairs


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ROOT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scene_summary(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["scene"].isin(["office", "redkitchen"])].copy()
    plot_df["recall_percent"] = plot_df["recall"] * 100.0

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("recall_percent", "Recall (%)"),
        ("mean_chamfer_distance", "Mean Chamfer Distance"),
        ("mean_rotation_error_deg", "Mean Rotation Error (°)"),
        ("mean_runtime_ms", "Mean Runtime (ms)"),
    ]
    palette = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        sns.barplot(data=plot_df, x="scene", y=metric, hue="method", palette=palette, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.legend_.set_title("Method")
        for container in ax.containers:
            labels = []
            for bar in container:
                h = bar.get_height()
                labels.append(f"{h:.2f}")
            ax.bar_label(container, labels=labels, padding=2, fontsize=9)

    save_fig(fig, "figure_hybrid_scene_summary.png")


def plot_pairwise_distributions(pairs: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    palette = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}
    metrics = [
        ("rotation_error_deg", "Rotation Error (°)"),
        ("translation_error_m", "Translation Error (m)"),
        ("chamfer_distance", "Chamfer Distance"),
        ("runtime_ms", "Runtime (ms)"),
    ]

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        sns.boxplot(data=pairs, x="method", y=metric, hue="scene", ax=ax)
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.legend_.set_title("Scene")

    save_fig(fig, "figure_hybrid_pair_distributions.png")


def plot_icp_iterations(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["scene"].isin(["office", "redkitchen"]) & summary["method"].isin(["A", "C"])].copy()
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"A": "#4C78A8", "C": "#54A24B"}
    sns.barplot(data=plot_df, x="scene", y="mean_icp_iterations", hue="method", palette=palette, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Mean ICP Iterations")
    ax.legend(title="Method")
    for container in ax.containers:
        labels = [f"{bar.get_height():.2f}" for bar in container]
        ax.bar_label(container, labels=labels, padding=2, fontsize=10)
    save_fig(fig, "figure_hybrid_icp_iterations.png")


def main() -> None:
    summary, pairs = load_data()
    plot_scene_summary(summary)
    plot_pairwise_distributions(pairs)
    plot_icp_iterations(summary)
    print(ROOT / "figure_hybrid_scene_summary.png")
    print(ROOT / "figure_hybrid_pair_distributions.png")
    print(ROOT / "figure_hybrid_icp_iterations.png")


if __name__ == "__main__":
    main()