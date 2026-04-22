"""
Create a 2x3 representative image grid from 7-Scenes, ScanNet, and KITTI.

Rows: 2 representative samples per dataset.
Cols: [7-Scenes, ScanNet, KITTI].
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
from PIL import Image


@dataclass
class CandidateImage:
    dataset: str
    path: Path
    score: float
    hist: np.ndarray


def setup_chinese_font() -> None:
    candidate_families = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans CN",
        "AR PL UMing CN",
    ]
    candidate_font_files = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    ]

    for font_file in candidate_font_files:
        try:
            if Path(font_file).exists():
                font_manager.fontManager.addfont(font_file)
        except Exception:
            continue

    available_families = {f.name for f in font_manager.fontManager.ttflist}
    selected_family = next((f for f in candidate_families if f in available_families), None)

    if selected_family is not None:
        rcParams["font.family"] = [selected_family]
        rcParams["font.sans-serif"] = [selected_family]
        print(f"✓ 中文字体已启用: {selected_family}")
    else:
        rcParams["font.family"] = ["DejaVu Sans"]
        rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("⚠️ 未检测到可用中文字体，中文可能显示为方块。")

    rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create representative dataset image grid (2x3)")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/hba/Documents/Dataset"),
        help="Root path that contains 7_scenes, ScanNet, KITTI",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("tests/tests_result/dataset_examples_2x3.png"),
        help="Output collage path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=1500,
        help="Max candidate images to evaluate per dataset",
    )
    return parser.parse_args()


def list_images_7scenes(root: Path) -> List[Path]:
    base = root / "7_scenes"
    return sorted(base.rglob("*.color.png"))


def list_images_scannet(root: Path) -> List[Path]:
    base = root / "ScanNet" / "scans"
    return sorted(base.glob("scene*/color/*.jpg"))


def list_images_kitti(root: Path) -> List[Path]:
    base = root / "KITTI" / "data_scene_flow" / "training" / "image_2"
    return sorted(base.glob("*.png"))


def sample_paths(paths: Sequence[Path], max_count: int, rng: random.Random) -> List[Path]:
    if len(paths) <= max_count:
        return list(paths)
    idx = sorted(rng.sample(range(len(paths)), max_count))
    return [paths[i] for i in idx]


def image_to_array(path: Path, max_side: int = 512) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    return np.array(image)


def compute_hist(image: np.ndarray, bins: int = 16) -> np.ndarray:
    hist_parts: List[np.ndarray] = []
    for channel in range(3):
        h, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 255), density=True)
        hist_parts.append(h)
    hist = np.concatenate(hist_parts).astype(np.float32)
    norm = np.linalg.norm(hist) + 1e-8
    return hist / norm


def compute_score(image: np.ndarray) -> float:
    gray = image.mean(axis=2).astype(np.float32)

    # Brightness confidence: avoid too dark / too bright
    brightness = gray.mean() / 255.0
    brightness_penalty = np.exp(-((brightness - 0.5) ** 2) / (2 * 0.18 * 0.18))

    # Contrast (texture richness)
    contrast = gray.std() / 64.0

    # Edge density via finite differences
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_strength = (gx + gy) / 2.0 / 32.0

    # Entropy
    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist + 1e-8
    entropy = -np.sum(hist * np.log(hist)) / np.log(64)

    score = 0.30 * contrast + 0.35 * edge_strength + 0.25 * entropy + 0.10 * brightness_penalty
    return float(score)


def choose_two_representatives(dataset: str, paths: Sequence[Path], rng: random.Random) -> List[CandidateImage]:
    candidates: List[CandidateImage] = []

    for path in paths:
        try:
            image = image_to_array(path)
            score = compute_score(image)
            hist = compute_hist(image)
            candidates.append(CandidateImage(dataset=dataset, path=path, score=score, hist=hist))
        except Exception:
            continue

    if len(candidates) < 2:
        raise RuntimeError(f"Not enough valid images for {dataset}. Found {len(candidates)}")

    candidates.sort(key=lambda item: item.score, reverse=True)
    first = candidates[0]

    pool = candidates[: min(120, len(candidates))]

    def second_objective(item: CandidateImage) -> float:
        diversity = np.linalg.norm(item.hist - first.hist)
        return 0.65 * item.score + 0.35 * diversity

    second = max(pool[1:], key=second_objective)
    return [first, second]


def draw_grid(selected: Dict[str, List[CandidateImage]], output_path: Path) -> None:
    datasets = ["7-Scenes", "ScanNet", "KITTI"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, dataset in enumerate(datasets):
        images = selected[dataset]
        for row in range(2):
            candidate = images[row]
            array = image_to_array(candidate.path, max_side=720)
            axes[row, col].imshow(array)
            axes[row, col].axis("off")

            short_name = candidate.path.name
            if dataset == "7-Scenes":
                subtitle = f"{candidate.path.parent.parent.name}/{candidate.path.parent.name}/{short_name}"
            elif dataset == "ScanNet":
                subtitle = f"{candidate.path.parent.parent.name}/{short_name}"
            else:
                subtitle = short_name

            axes[row, col].set_title(f"{dataset} - 示例{row + 1}\n{subtitle}", fontsize=10)

    fig.suptitle("代表性数据集示例图（两行三列）", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    rng = random.Random(args.seed)

    mapping = {
        "7-Scenes": list_images_7scenes(args.dataset_root),
        "ScanNet": list_images_scannet(args.dataset_root),
        "KITTI": list_images_kitti(args.dataset_root),
    }

    selected: Dict[str, List[CandidateImage]] = {}
    for dataset, paths in mapping.items():
        if not paths:
            raise RuntimeError(f"No images found for dataset: {dataset}")
        sampled = sample_paths(paths, args.max_samples_per_dataset, rng)
        selected[dataset] = choose_two_representatives(dataset, sampled, rng)

    draw_grid(selected, args.output_path)

    print("=" * 90)
    print("✅ 2x3 representative grid generated")
    print(f"Output: {args.output_path}")
    for dataset, picks in selected.items():
        print(f"{dataset}:")
        for idx, pick in enumerate(picks, start=1):
            print(f"  [{idx}] {pick.path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
