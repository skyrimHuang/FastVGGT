import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Table 3.1 benchmark with incremental CSV writing and multi-scene averaging"
    )
    parser.add_argument("--image_dir", type=str, default="assets", help="Input image directory")
    parser.add_argument("--size", type=int, default=518, help="Target square input size")
    parser.add_argument("--num_scenes", type=int, default=10, help="Number of images/scenes to evaluate")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per scene")
    parser.add_argument(
        "--runs_per_scene",
        type=int,
        default=1,
        help="Measured runs per method for each scene (keep small for speed)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/model_tracker_fixed_e20.pt",
        help="VGGT checkpoint",
    )
    parser.add_argument(
        "--dino_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="DINOv2 device",
    )
    parser.add_argument(
        "--vggt_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="VGGT device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/tests_result/table3_1",
        help="Output directory",
    )
    parser.add_argument(
        "--surf_hessian_threshold",
        type=float,
        default=500.0,
        help="SURF hessian threshold",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_scene_images(image_dir: Path, num_scenes: int) -> List[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return paths[: max(1, min(num_scenes, len(paths)))]


def load_scene_inputs(image_path: Path, size: int) -> Tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
    return gray, tensor


def cuda_sync_if_needed(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(
    fn: Callable[[], None],
    warmup: int,
    runs: int,
    sync_device: Optional[torch.device],
) -> float:
    for _ in range(max(0, warmup)):
        fn()

    elapsed = []
    for _ in range(max(1, runs)):
        cuda_sync_if_needed(sync_device)
        start = time.perf_counter()
        fn()
        cuda_sync_if_needed(sync_device)
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(elapsed))


def build_dinov2(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


def build_vggt(ckpt_path: str, device: torch.device) -> VGGT:
    model = VGGT(
        merging=999,
        merge_ratio=0.0,
        enable_point=True,
        enable_depth=True,
        enable_camera=True,
    ).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    model.update_patch_dimensions(37, 37)
    model.to(device)
    return model


def append_row(csv_path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def safe_fps(latency_ms: Optional[float]) -> Optional[float]:
    if latency_ms is None or latency_ms <= 0:
        return None
    return 1000.0 / latency_ms


def update_summary(detail_csv: Path, summary_csv: Path, summary_md: Path) -> pd.DataFrame:
    df = pd.read_csv(detail_csv)
    success_df = df[df["success"] == True].copy()  # noqa: E712

    rows = []
    method_order = [
        "SIFT",
        "SURF",
        "ORB",
        "AKAZE",
        "BRISK",
        "DINOv2-ViT-S/14",
        "VGGT",
    ]

    for method in method_order:
        method_all = df[df["算法模块/模型"] == method]
        method_ok = success_df[success_df["算法模块/模型"] == method]
        if len(method_ok) > 0:
            mean_ms = float(method_ok["耗时(ms)"].mean())
            fps = safe_fps(mean_ms)
            rows.append(
                {
                    "算法模块/模型": method,
                    "耗时(ms)": mean_ms,
                    "处理帧率(FPS)": fps,
                    "成功场景数": int(len(method_ok)),
                    "总场景数": int(len(method_all)),
                }
            )
        elif len(method_all) > 0:
            rows.append(
                {
                    "算法模块/模型": method,
                    "耗时(ms)": np.nan,
                    "处理帧率(FPS)": np.nan,
                    "成功场景数": 0,
                    "总场景数": int(len(method_all)),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    md_lines = ["表3.1 不同算法模块的单帧处理耗时与帧率对比（多场景平均）", ""]
    md_lines.append("| 算法模块/模型 | 耗时(ms) | 处理帧率(FPS) | 成功场景数 | 总场景数 |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        ms_text = f"{row['耗时(ms)']:.1f}" if pd.notna(row["耗时(ms)"]) else "N/A"
        fps_text = f"{row['处理帧率(FPS)']:.1f}" if pd.notna(row["处理帧率(FPS)"]) else "N/A"
        md_lines.append(
            f"| {row['算法模块/模型']} | {ms_text} | {fps_text} | {int(row['成功场景数'])} | {int(row['总场景数'])} |"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return summary_df


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    scenes = list_scene_images(image_dir, args.num_scenes)

    output_dir = Path(args.output_dir)
    detail_csv = output_dir / "table3_1_single_frame_detail_incremental.csv"
    summary_csv = output_dir / "table3_1_single_frame_summary.csv"
    summary_md = output_dir / "table3_1_single_frame.md"

    if detail_csv.exists():
        detail_csv.unlink()

    dino_device = resolve_device(args.dino_device)
    vggt_device = resolve_device(args.vggt_device)

    dino_model: Optional[torch.nn.Module] = None
    vggt_model: Optional[VGGT] = None

    dino_error: Optional[str] = None
    vggt_error: Optional[str] = None

    try:
        dino_model = build_dinov2(dino_device)
    except Exception as error:
        dino_error = str(error)

    try:
        if not Path(args.ckpt_path).exists():
            raise FileNotFoundError(f"VGGT checkpoint not found: {args.ckpt_path}")
        vggt_model = build_vggt(args.ckpt_path, vggt_device)
    except Exception as error:
        vggt_error = str(error)

    fieldnames = [
        "scene_idx",
        "image_name",
        "算法模块/模型",
        "耗时(ms)",
        "处理帧率(FPS)",
        "device",
        "success",
        "error_msg",
        "size",
        "warmup",
        "runs_per_scene",
    ]

    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()

    surf_obj = None
    surf_error: Optional[str] = None
    try:
        if not hasattr(cv2, "xfeatures2d"):
            raise RuntimeError("OpenCV xfeatures2d is unavailable")
        surf_obj = cv2.xfeatures2d.SURF_create(hessianThreshold=args.surf_hessian_threshold)
    except Exception as error:
        surf_error = str(error)

    def write_result(
        scene_idx: int,
        image_name: str,
        method: str,
        latency_ms: Optional[float],
        device_name: str,
        success: bool,
        error_msg: str,
    ) -> None:
        row = {
            "scene_idx": scene_idx,
            "image_name": image_name,
            "算法模块/模型": method,
            "耗时(ms)": latency_ms if latency_ms is not None else np.nan,
            "处理帧率(FPS)": safe_fps(latency_ms) if latency_ms is not None else np.nan,
            "device": device_name,
            "success": success,
            "error_msg": error_msg,
            "size": args.size,
            "warmup": args.warmup,
            "runs_per_scene": args.runs_per_scene,
        }
        append_row(detail_csv, row, fieldnames)

    for scene_idx, image_path in enumerate(scenes, start=1):
        gray, rgb_tensor = load_scene_inputs(image_path, args.size)

        methods_cpu: List[Tuple[str, Callable[[], None]]] = [
            ("SIFT", lambda: sift.detectAndCompute(gray, None)),
            ("ORB", lambda: orb.detectAndCompute(gray, None)),
            ("AKAZE", lambda: akaze.detectAndCompute(gray, None)),
            ("BRISK", lambda: brisk.detectAndCompute(gray, None)),
        ]

        for method_name, method_fn in methods_cpu:
            try:
                latency = measure_ms(method_fn, args.warmup, args.runs_per_scene, sync_device=None)
                write_result(scene_idx, image_path.name, method_name, latency, "cpu", True, "")
            except Exception as error:
                write_result(scene_idx, image_path.name, method_name, None, "cpu", False, str(error))

        if surf_obj is not None:
            try:
                latency = measure_ms(
                    lambda: surf_obj.detectAndCompute(gray, None),
                    args.warmup,
                    args.runs_per_scene,
                    sync_device=None,
                )
                write_result(scene_idx, image_path.name, "SURF", latency, "cpu", True, "")
            except Exception as error:
                write_result(scene_idx, image_path.name, "SURF", None, "cpu", False, str(error))
        else:
            write_result(
                scene_idx,
                image_path.name,
                "SURF",
                None,
                "cpu",
                False,
                surf_error or "SURF unavailable",
            )

        if dino_model is not None:
            try:
                dino_input = rgb_tensor.to(dino_device)

                def dino_fn() -> None:
                    with torch.no_grad():
                        _ = dino_model(dino_input)

                latency = measure_ms(
                    dino_fn,
                    args.warmup,
                    args.runs_per_scene,
                    sync_device=dino_device if dino_device.type == "cuda" else None,
                )
                write_result(
                    scene_idx,
                    image_path.name,
                    "DINOv2-ViT-S/14",
                    latency,
                    str(dino_device),
                    True,
                    "",
                )
            except Exception as error:
                write_result(
                    scene_idx,
                    image_path.name,
                    "DINOv2-ViT-S/14",
                    None,
                    str(dino_device),
                    False,
                    str(error),
                )
        else:
            write_result(
                scene_idx,
                image_path.name,
                "DINOv2-ViT-S/14",
                None,
                str(dino_device),
                False,
                dino_error or "DINOv2 unavailable",
            )

        if vggt_model is not None:
            try:
                vggt_input = rgb_tensor.to(vggt_device)

                def vggt_fn() -> None:
                    with torch.no_grad():
                        if vggt_device.type == "cuda":
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                _ = vggt_model(vggt_input)
                        else:
                            _ = vggt_model(vggt_input)

                latency = measure_ms(
                    vggt_fn,
                    args.warmup,
                    args.runs_per_scene,
                    sync_device=vggt_device if vggt_device.type == "cuda" else None,
                )
                write_result(
                    scene_idx,
                    image_path.name,
                    "VGGT",
                    latency,
                    str(vggt_device),
                    True,
                    "",
                )
            except Exception as error:
                write_result(scene_idx, image_path.name, "VGGT", None, str(vggt_device), False, str(error))
        else:
            write_result(
                scene_idx,
                image_path.name,
                "VGGT",
                None,
                str(vggt_device),
                False,
                vggt_error or "VGGT unavailable",
            )

    summary_df = update_summary(detail_csv, summary_csv, summary_md)

    display_df = summary_df.copy()
    if not display_df.empty:
        display_df["耗时(ms)"] = display_df["耗时(ms)"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        display_df["处理帧率(FPS)"] = display_df["处理帧率(FPS)"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

    print("\n===== Table 3.1 (multi-scene average) =====")
    if display_df.empty:
        print("No results generated.")
    else:
        print(display_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- Incremental detail CSV : {detail_csv}")
    print(f"- Summary CSV            : {summary_csv}")
    print(f"- Markdown table         : {summary_md}")


if __name__ == "__main__":
    main()
