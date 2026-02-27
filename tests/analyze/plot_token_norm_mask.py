import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from matplotlib import font_manager
import torch

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.utils.eval_utils import get_vgg_input_imgs


# 设置中文字体
def setup_chinese_font():
    """配置中文字体支持，参考 plot_module_latency.py"""
    font_candidates = [
        "simhei",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "Microsoft YaHei",
    ]
    
    for font in font_candidates:
        try:
            if Path(font).exists():
                font_manager.fontManager.addfont(font)
                font_prop = font_manager.FontProperties(fname=font)
                plt.rcParams['font.family'] = font_prop.get_name()
                break
        except:
            continue
    
    # 如果找不到中文字体，使用默认
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return font_manager.FontProperties(family=plt.rcParams['font.family'])


FONT_PROP = setup_chinese_font()


def load_rgb_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(ckpt_path: Path, device: torch.device) -> VGGT:
    model = VGGT(merging=0, merge_ratio=0.9, vis_attn_map=False, use_norm_guided=True)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    return model


def compute_patch_norms(
    model: VGGT,
    vgg_input: torch.Tensor,
    device: torch.device,
    use_norm1: bool,
) -> torch.Tensor:
    """Return L2 norms for patch tokens as a flat tensor [P]."""
    agg = model.aggregator
    patch_start_idx = agg.patch_start_idx

    with torch.no_grad():
        images = vgg_input.unsqueeze(0).to(device)  # [B=1, S, 3, H, W]
        images = images.to(torch.float32)
        images = (images - agg._resnet_mean.to(device)) / agg._resnet_std.to(device)

        bsz, seq, _, h, w = images.shape
        images = images.view(bsz * seq, 3, h, w)
        patch_tokens = agg.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        camera_token = slice_expand_and_flatten(agg.camera_token.to(device), bsz, seq)
        register_token = slice_expand_and_flatten(agg.register_token.to(device), bsz, seq)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        if use_norm1:
            tokens = agg.frame_blocks[0].norm1(tokens)

        patch_tokens = tokens[:, patch_start_idx:, :]
        norms = torch.norm(patch_tokens.float(), dim=-1)

    return norms.squeeze(0)


def split_tokens_by_norm(
    norms: torch.Tensor,
    patch_height: int,
    patch_width: int,
    salient_ratio: float,
    dst_ratio: float,
) -> np.ndarray:
    """Norm-guided split: use L2 norm to identify important tokens."""
    num_patches = patch_height * patch_width
    flat = norms.flatten()

    sorted_idx = torch.argsort(flat, descending=True)
    num_salient = max(1, int(num_patches * salient_ratio))
    num_dst = max(1, int(num_patches * dst_ratio))

    labels = torch.zeros(num_patches, dtype=torch.int64)
    labels[sorted_idx[:num_salient]] = 2  # salient
    labels[sorted_idx[num_salient : num_salient + num_dst]] = 1  # dst

    return labels.view(patch_height, patch_width).cpu().numpy()


def split_tokens_by_grid(
    patch_height: int,
    patch_width: int,
    sx: int = 2,
    sy: int = 2,
    enable_protection: bool = True,
    protection_ratio: float = 0.10,
) -> np.ndarray:
    """Grid-based split (baseline): uniform spatial sampling without using norms.
    
    Divides the patch grid into blocks of size sy×sx, selects top-left token 
    in each block as dst (anchor), others as src (to be merged).
    
    If enable_protection=True, uniformly samples protection_ratio of tokens 
    as salient (protected) using stride-based sampling.
    
    Args:
        patch_height: Height of the patch grid
        patch_width: Width of the patch grid
        sx: Block stride in x dimension
        sy: Block stride in y dimension
        enable_protection: If True, protect some tokens as salient
        protection_ratio: Ratio of tokens to protect (default 0.10)
    
    Returns:
        labels: 2D array where 0=src, 1=dst, 2=salient
    """
    num_patches = patch_height * patch_width
    labels = np.zeros((patch_height, patch_width), dtype=np.int64)  # default: src (0)
    
    # Step 1: Mark salient tokens (protected) using uniform stride sampling
    if enable_protection:
        num_protected = int(num_patches * protection_ratio)
        step = max(1, num_patches // num_protected)
        
        # Generate uniform indices
        protected_flat_indices = np.arange(0, num_patches, step)[:num_protected]
        
        # Convert flat indices to 2D coordinates
        for idx in protected_flat_indices:
            y = idx // patch_width
            x = idx % patch_width
            if y < patch_height:
                labels[y, x] = 2  # salient (protected)
    
    # Step 2: Mark dst tokens using grid-based split
    # Calculate number of blocks
    hsy = patch_height // sy
    wsx = patch_width // sx
    
    # Mark the top-left token of each block as dst (only if not already salient)
    for by in range(hsy):
        for bx in range(wsx):
            # Top-left corner of the block
            y_start = by * sy
            x_start = bx * sx
            if labels[y_start, x_start] != 2:  # Don't override salient tokens
                labels[y_start, x_start] = 1  # dst
    
    return labels


def upsample_map(map_2d: np.ndarray, target_size: tuple, mode: str) -> np.ndarray:
    pil_mode = "L" if map_2d.dtype != np.float32 else "F"
    img = Image.fromarray(map_2d, mode=pil_mode)
    resample = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    img = img.resize(target_size, resample=resample)
    return np.array(img)


def normalize_01(values: np.ndarray) -> np.ndarray:
    vmin = values.min()
    vmax = values.max()
    return (values - vmin) / (vmax - vmin + 1e-8)


def plot_figure(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    mask_rgb_baseline: np.ndarray,
    mask_rgb_normguided: np.ndarray,
    output_path: Path,
):
    """绘制4列对比图：RGB | L2范数热力图 | Baseline分割 | Norm-Guided分割"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)

    # 第1列：原始RGB图
    axes[0].imshow(image_np)
    axes[0].set_title("RGB 图像", fontsize=12, fontproperties=FONT_PROP, fontweight='bold')
    axes[0].axis("off")

    # 第2列：Token L2范数热力图
    axes[1].imshow(image_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.55)
    axes[1].set_title("Token L2 范数", fontsize=12, fontproperties=FONT_PROP, fontweight='bold')
    axes[1].axis("off")

    # 第3列：Baseline划分掩码（网格固定采样）
    axes[2].imshow(mask_rgb_baseline)
    axes[2].set_title("基线方法（网格采样）", fontsize=12, fontproperties=FONT_PROP, fontweight='bold')
    axes[2].axis("off")

    # 第4列：Norm-Guided划分掩码（TopK范数）
    axes[3].imshow(mask_rgb_normguided)
    axes[3].set_title("本文方法（范数引导）", fontsize=12, fontproperties=FONT_PROP, fontweight='bold')
    axes[3].axis("off")

    # 在底部添加统一的图例
    legend_items = [
        mpatches.Patch(color=(255/255, 0/255, 0/255), label="显著Token"),
        mpatches.Patch(color=(0/255, 0/255, 255/255), label="目标Token"),
        mpatches.Patch(color=(0/255, 255/255, 0/255), label="源Token"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=11,
        prop=FONT_PROP,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("./tests/tests_result/fig3_11_token_norm_mask.png"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("./ckpt/model_tracker_fixed_e20.pt"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--salient_ratio", type=float, default=0.10)
    parser.add_argument("--dst_ratio", type=float, default=0.40)
    parser.add_argument(
        "--grid_sx", type=int, default=2, help="Grid block width for baseline grid-based split"
    )
    parser.add_argument(
        "--grid_sy", type=int, default=2, help="Grid block height for baseline grid-based split"
    )
    parser.add_argument(
        "--no_norm1",
        action="store_true",
        help="Disable the first block LayerNorm before computing norms",
    )
    parser.add_argument(
        "--no_protection",
        action="store_true",
        help="Disable salient token protection (no red tokens in visualization)",
    )
    parser.add_argument(
        "--protection_ratio",
        type=float,
        default=0.10,
        help="Ratio of tokens to protect as salient (default: 0.10)",
    )
    args = parser.parse_args()

    device = select_device(args.device)

    image_np = load_rgb_image(args.image_path)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(
        np.stack([image_np], axis=0)
    )

    model = build_model(args.ckpt_path, device)
    norms = compute_patch_norms(model, vgg_input, device, not args.no_norm1)

    norm_map = norms.view(patch_height, patch_width).cpu().numpy()
    norm_map = normalize_01(norm_map.astype(np.float32))

    resized_img = vgg_input[0].permute(1, 2, 0).cpu().numpy()
    resized_img = (resized_img * 255.0).clip(0, 255).astype(np.uint8)

    heatmap = upsample_map(norm_map, (resized_img.shape[1], resized_img.shape[0]), "bilinear")
    
    enable_protection = not args.no_protection
    
    # 生成 Baseline (Grid-Based) 分割
    labels_baseline = split_tokens_by_grid(
        patch_height, patch_width, args.grid_sx, args.grid_sy,
        enable_protection=enable_protection,
        protection_ratio=args.protection_ratio
    )
    labels_baseline_up = upsample_map(
        labels_baseline.astype(np.uint8), 
        (resized_img.shape[1], resized_img.shape[0]), 
        "nearest"
    )
    
    # 生成 Norm-Guided (TopK) 分割
    if enable_protection:
        labels_normguided = split_tokens_by_norm(
            norms, patch_height, patch_width, args.salient_ratio, args.dst_ratio
        )
    else:
        # If protection disabled, only have dst and src
        labels_normguided = split_tokens_by_norm(
            norms, patch_height, patch_width, 0.0, args.dst_ratio + args.salient_ratio
        )
    labels_normguided_up = upsample_map(
        labels_normguided.astype(np.uint8), 
        (resized_img.shape[1], resized_img.shape[0]), 
        "nearest"
    )

    # 颜色映射：红=salient, 蓝=dst, 绿=src
    colors = np.array(
        [
            [0, 255, 0],    # src (0) = 绿色
            [0, 0, 255],    # dst (1) = 蓝色
            [255, 0, 0],    # salient (2) = 红色
        ],
        dtype=np.uint8,
    )
    
    mask_rgb_baseline = colors[labels_baseline_up]
    mask_rgb_normguided = colors[labels_normguided_up]

    plot_figure(resized_img, heatmap, mask_rgb_baseline, mask_rgb_normguided, args.output_path)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
