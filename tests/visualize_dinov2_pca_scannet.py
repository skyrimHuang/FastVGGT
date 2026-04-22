"""
ScanNet + DINOv2 + Global PCA-to-RGB visualization.

Workflow:
1) Randomly select N scenes from ScanNet.
2) For each scene, sample K frames with stride S (e.g., every 3 frames).
3) Extract DINOv2 patch-token features.
4) Fit a GLOBAL PCA (3 components) over all sampled patch features.
5) Map PCA(3D) to RGB and save visualizations.

Expected effect:
Parts with similar semantic/geometric properties should show consistent colors
across different images.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


@dataclass
class FrameSample:
    scene_id: str
    frame_id: int
    image_path: Path


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
    rcParams["font.size"] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ScanNet semantics via DINOv2 + PCA RGB")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/hba/Documents/Dataset/ScanNet/scans"),
        help="ScanNet scans root directory",
    )
    parser.add_argument(
        "--scene_list_yaml",
        type=Path,
        default=Path("eval/scannet_50.yaml"),
        help="Optional scene list yaml/text (one scene id per line)",
    )
    parser.add_argument("--num_scenes", type=int, default=5, help="How many random scenes to sample")
    parser.add_argument("--frames_per_scene", type=int, default=5, help="Frames sampled per scene")
    parser.add_argument("--frame_step", type=int, default=3, help="Stride between sampled frames")
    parser.add_argument("--seed", type=int, default=33, help="Random seed")
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINOv2 model from torch.hub",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=518,
        help="Model input size (must be divisible by 14 for DINOv2 patch grid)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tests/tests_result/dinov2_pca_semantic_consistency"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_scene_candidates(data_dir: Path, scene_list_yaml: Path) -> List[str]:
    if scene_list_yaml.exists():
        with open(scene_list_yaml, "r", encoding="utf-8") as file:
            listed = [line.strip() for line in file if line.strip()]
        candidates = [scene for scene in listed if (data_dir / scene).is_dir()]
        if candidates:
            return candidates

    return sorted([directory.name for directory in data_dir.iterdir() if directory.is_dir()])


def get_frame_id_to_path(color_dir: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in color_dir.glob(ext):
            try:
                mapping[int(path.stem)] = path
            except ValueError:
                continue
    return mapping


def get_valid_frame_ids(scene_dir: Path) -> Tuple[List[int], Dict[int, Path]]:
    color_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"

    if not color_dir.is_dir() or not pose_dir.is_dir():
        return [], {}

    frame_to_path = get_frame_id_to_path(color_dir)
    color_ids = set(frame_to_path.keys())

    pose_ids: set[int] = set()
    for pose_file in pose_dir.glob("*.txt"):
        try:
            pose_ids.add(int(pose_file.stem))
        except ValueError:
            continue

    valid_ids = sorted(list(color_ids & pose_ids))
    return valid_ids, frame_to_path


def sample_ids_with_stride(
    valid_ids: Sequence[int],
    frames_per_scene: int,
    frame_step: int,
    rng: random.Random,
) -> List[int]:
    needed_span = 1 + (frames_per_scene - 1) * frame_step
    if len(valid_ids) < needed_span:
        return []

    max_start = len(valid_ids) - needed_span
    start_idx = rng.randint(0, max_start)
    return [valid_ids[start_idx + i * frame_step] for i in range(frames_per_scene)]


def sample_scenes_and_frames(
    data_dir: Path,
    candidate_scenes: Sequence[str],
    num_scenes: int,
    frames_per_scene: int,
    frame_step: int,
    seed: int,
) -> List[FrameSample]:
    rng = random.Random(seed)

    eligible: List[Tuple[str, List[int], Dict[int, Path]]] = []
    for scene_id in candidate_scenes:
        scene_dir = data_dir / scene_id
        valid_ids, frame_to_path = get_valid_frame_ids(scene_dir)
        sampled_ids = sample_ids_with_stride(valid_ids, frames_per_scene, frame_step, rng)
        if sampled_ids:
            eligible.append((scene_id, sampled_ids, frame_to_path))

    if len(eligible) < num_scenes:
        raise RuntimeError(
            f"Only {len(eligible)} eligible scenes found, but num_scenes={num_scenes}."
        )

    chosen = rng.sample(eligible, num_scenes)
    samples: List[FrameSample] = []
    for scene_id, sampled_ids, frame_to_path in chosen:
        for frame_id in sampled_ids:
            samples.append(
                FrameSample(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    image_path=frame_to_path[frame_id],
                )
            )
    return samples


def build_preprocess(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load DINOv2 from torch.hub. Ensure internet/cache availability and rerun."
        ) from exc

    model = model.to(device)
    model.eval()
    return model


def extract_patch_tokens(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            feats = model.forward_features(batch)
            if isinstance(feats, dict):
                if "x_norm_patchtokens" in feats:
                    return feats["x_norm_patchtokens"]
                if "x_prenorm" in feats:
                    tokens = feats["x_prenorm"]
                    if tokens.ndim == 3:
                        return tokens[:, 1:, :]
            elif torch.is_tensor(feats):
                return feats

        if hasattr(model, "get_intermediate_layers"):
            layers = model.get_intermediate_layers(batch, n=1)
            tokens = layers[0]
            if tokens.ndim == 3:
                return tokens[:, 1:, :]

    raise RuntimeError("Unable to extract patch tokens from DINOv2 model output.")


def pca_project_to_rgb(features: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T  # [D, 3]
    projected = centered @ components  # [N, 3]

    explained_variance = (singular_values**2) / max(features.shape[0] - 1, 1)
    explained_ratio = explained_variance[:3] / (explained_variance.sum() + 1e-12)

    rgb = np.zeros_like(projected, dtype=np.float32)
    for channel in range(3):
        low = np.percentile(projected[:, channel], 1)
        high = np.percentile(projected[:, channel], 99)
        denom = max(high - low, 1e-6)
        rgb[:, channel] = np.clip((projected[:, channel] - low) / denom, 0.0, 1.0)

    return rgb, explained_ratio.tolist()


def save_side_by_side(original: np.ndarray, pca_rgb: np.ndarray, out_path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("原图")
    axes[0].axis("off")

    axes[1].imshow(pca_rgb)
    axes[1].set_title("DINOv2+PCA伪彩色")
    axes[1].axis("off")

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def save_scene_grid(
    scene_id: str,
    originals: List[np.ndarray],
    pca_rgbs: List[np.ndarray],
    frame_ids: List[int],
    out_path: Path,
) -> None:
    num_frames = len(originals)
    figure, axes = plt.subplots(2, num_frames, figsize=(3.2 * num_frames, 6.2))

    for idx in range(num_frames):
        axes[0, idx].imshow(originals[idx])
        axes[0, idx].set_title(f"Frame {frame_ids[idx]} 原图", fontsize=9)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(pca_rgbs[idx])
        axes[1, idx].set_title(f"Frame {frame_ids[idx]} PCA-RGB", fontsize=9)
        axes[1, idx].axis("off")

    figure.suptitle(f"{scene_id}: 每隔3帧采样的跨视角语义颜色一致性", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=240)
    plt.close(figure)


def save_overview_grid(
    scene_to_original_ref: Dict[str, np.ndarray],
    scene_to_pca: Dict[str, List[np.ndarray]],
    scene_to_frame_ids: Dict[str, List[int]],
    out_path: Path,
) -> None:
    scene_ids = list(scene_to_pca.keys())
    rows = len(scene_ids)
    pca_cols = len(next(iter(scene_to_pca.values())))
    cols = pca_cols + 1  # leftmost column is original RGB reference

    figure, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.7 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, scene_id in enumerate(scene_ids):
        frames = scene_to_frame_ids[scene_id]
        axes[row, 0].imshow(scene_to_original_ref[scene_id])
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("RGB参考", fontsize=9)
        axes[row, 0].set_ylabel(scene_id, fontsize=9)

        for col in range(pca_cols):
            axes[row, col + 1].imshow(scene_to_pca[scene_id][col])
            axes[row, col + 1].axis("off")
            if row == 0:
                axes[row, col + 1].set_title(f"PCA F{frames[col]}", fontsize=9)

    figure.suptitle("5个ScanNet序列总览：左列RGB参考 + 右侧PCA-RGB一致性", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=240)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    if args.input_size % 14 != 0:
        raise ValueError("--input_size must be divisible by 14 for DINOv2 patch tokens.")

    set_seed(args.seed)
    start_time = time.time()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_scenes = read_scene_candidates(args.data_dir, args.scene_list_yaml)
    if not candidate_scenes:
        raise RuntimeError(f"No scene candidates found under {args.data_dir}")

    samples = sample_scenes_and_frames(
        data_dir=args.data_dir,
        candidate_scenes=candidate_scenes,
        num_scenes=args.num_scenes,
        frames_per_scene=args.frames_per_scene,
        frame_step=args.frame_step,
        seed=args.seed,
    )

    preprocess = build_preprocess(args.input_size)
    model = load_dinov2(args.model_name, device)

    # Keep deterministic ordering: by scene then frame
    samples = sorted(samples, key=lambda item: (item.scene_id, item.frame_id))

    originals: List[np.ndarray] = []
    tensors: List[torch.Tensor] = []
    scene_to_indices: Dict[str, List[int]] = {}
    scene_to_frame_ids: Dict[str, List[int]] = {}

    for index, sample in enumerate(samples):
        image = Image.open(sample.image_path).convert("RGB")
        tensor = preprocess(image)
        resized_rgb = np.array(image.resize((args.input_size, args.input_size), Image.Resampling.BICUBIC))

        originals.append(resized_rgb)
        tensors.append(tensor)

        scene_to_indices.setdefault(sample.scene_id, []).append(index)
        scene_to_frame_ids.setdefault(sample.scene_id, []).append(sample.frame_id)

    batch = torch.stack(tensors, dim=0).to(device)
    patch_tokens = extract_patch_tokens(model, batch)  # [B, num_patches, C]

    patch_h = args.input_size // 14
    patch_w = args.input_size // 14
    expected_patches = patch_h * patch_w

    if patch_tokens.shape[1] != expected_patches:
        raise RuntimeError(
            f"Patch token count mismatch: got {patch_tokens.shape[1]}, expected {expected_patches}."
        )

    token_features = patch_tokens.detach().cpu().numpy()  # [B, P, C]
    all_features = token_features.reshape(-1, token_features.shape[-1])

    rgb_projected, explained_ratio = pca_project_to_rgb(all_features)
    rgb_per_image = rgb_projected.reshape(token_features.shape[0], expected_patches, 3)
    rgb_per_image = rgb_per_image.reshape(token_features.shape[0], patch_h, patch_w, 3)

    scene_to_pca_images: Dict[str, List[np.ndarray]] = {}
    scene_to_original_ref: Dict[str, np.ndarray] = {}

    for scene_id, indices in scene_to_indices.items():
        scene_dir = args.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene_originals: List[np.ndarray] = []
        scene_pca_rgbs: List[np.ndarray] = []

        for local_i, global_idx in enumerate(indices):
            frame_id = scene_to_frame_ids[scene_id][local_i]
            orig = originals[global_idx]

            pca_patch_rgb = (rgb_per_image[global_idx] * 255.0).astype(np.uint8)
            pca_img = np.array(
                Image.fromarray(pca_patch_rgb).resize(
                    (args.input_size, args.input_size), Image.Resampling.NEAREST
                )
            )

            Image.fromarray(orig).save(scene_dir / f"frame_{frame_id:06d}_rgb.png")
            Image.fromarray(pca_img).save(scene_dir / f"frame_{frame_id:06d}_pca_rgb.png")

            save_side_by_side(
                original=orig,
                pca_rgb=pca_img,
                out_path=scene_dir / f"frame_{frame_id:06d}_side_by_side.png",
                title=f"{scene_id} / frame {frame_id}",
            )

            scene_originals.append(orig)
            scene_pca_rgbs.append(pca_img)

        save_scene_grid(
            scene_id=scene_id,
            originals=scene_originals,
            pca_rgbs=scene_pca_rgbs,
            frame_ids=scene_to_frame_ids[scene_id],
            out_path=scene_dir / "scene_grid.png",
        )

        scene_to_pca_images[scene_id] = scene_pca_rgbs
        scene_to_original_ref[scene_id] = scene_originals[0]

    save_overview_grid(
        scene_to_original_ref=scene_to_original_ref,
        scene_to_pca=scene_to_pca_images,
        scene_to_frame_ids=scene_to_frame_ids,
        out_path=args.output_dir / "all_scenes_pca_overview.png",
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(args.data_dir),
        "scene_list_yaml": str(args.scene_list_yaml),
        "num_scenes": args.num_scenes,
        "frames_per_scene": args.frames_per_scene,
        "frame_step": args.frame_step,
        "seed": args.seed,
        "model_name": args.model_name,
        "input_size": args.input_size,
        "patch_grid": [patch_h, patch_w],
        "pca_explained_variance_ratio": explained_ratio,
        "selected_scenes": {
            scene_id: scene_to_frame_ids[scene_id] for scene_id in scene_to_frame_ids
        },
        "runtime_seconds": round(time.time() - start_time, 3),
    }

    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("✅ DINOv2 + PCA 语义一致性可视化完成")
    print(f"输出目录: {args.output_dir}")
    print(f"选中场景数: {len(scene_to_frame_ids)}")
    print(f"每场景帧数: {args.frames_per_scene} (步长={args.frame_step})")
    print(f"PCA解释方差比(前三主成分): {explained_ratio}")
    print("=" * 90)


if __name__ == "__main__":
    main()
