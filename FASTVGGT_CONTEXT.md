# FastVGGT Model Context

This document serves as a technical reference for the **FastVGGT** (Fast Visual Geometry Transformer) model. It connects the core innovations described in the paper with their specific code implementations, facilitating future optimizations and modifications.

## 1. Project Overview

**FastVGGT** is a training-free, efficiency-optimized framework designed to accelerate the inference of Visual Geometry Transformers (VGGT) for long-sequence 3D reconstruction tasks.

- **Core Goal**: Accelerate inference (up to 4x on 1000 frames) and reduce VRAM usage while maintaining reconstruction quality.
- **Key Method**: A specialized **Token Merging Strategy** tailored for multi-view geometry, avoiding "token collapse" in global attention layers.
- **Nature**: Plug-and-play; requires no re-training or fine-tuning of the original VGGT weights.

## 2. Core Innovations & Code Implementation

The following table maps the paper's innovations to the codebase.

### 2.1 Token Partitioning Strategy
The paper proposes a three-part token partitioning strategy to ensure geometric consistency and detail preservation.

| Strategy Component | Paper Description | Code Implementation (`merging/merge.py`) |
|-------------------|-------------------|------------------------------------------|
| **Reference Tokens** | Keep initial frame tokens as global reference. | **Lines 104-106**: `idx_buffer_seq[:tokens_per_img] = -1`. The first frame is explicitly marked as `dst` (destination), ensuring it is never merged. |
| **Salient Tokens** | Keep top 10% tokens with high importance. | **Lines 90-95, 230-241**: Controlled by `enable_protection=True`. Calculates `num_protected` (10%) and filters `src_indices` to exclude protected tokens from being merged. |
| **Uniform Sampling** | Region-based random sampling for spatial balance. | **Lines 134-157**: Uses grid-based sampling (`sx`, `sy`) to select `dst` tokens. `idx_buffer_batch.scatter_` ensures one `dst` token per grid region. |

### 2.2 Token Merging & Unmerging
To maintain compatibility with the standard Transformer architecture (specifically Frame Attention), tokens are merged before Global Attention and unmerged afterwards.

*   **Location**: `vggt/layers/attention.py` inside `Attention.forward`.
*   **Merging Process**:
    1.  **Trigger**: Checks `if global_merging is not None` (Line 167).
    2.  **Partitioning**: Calls `token_merge_bipartite2d` (Line 173) with `enable_protection=True`.
    3.  **Execution**: Calls `m_a` (merge function) on Q, K, V (Lines 193-198) to reduce sequence length ($N \to N_m$).
    4.  **Attention**: Computes attention on reduced tokens (Lines 211-216).
*   **Unmerging Process**:
    1.  **Execution**: Calls `u_a` (unmerge function) on the output `x` (Line 223).
    2.  **Result**: Restores the original sequence length $N$, filling merged positions with the result of their corresponding `dst` token.

### 2.3 Efficiency & Optimization
*   **Similarity Computation**: `fast_similarity_chunks` in `merging/merge.py` uses BF16 and chunking to speed up the calculation of cosine similarity between tokens.
*   **VRAM Optimization**: By reducing the number of tokens in the quadratic attention operation ($O(N^2) \to O(N_m^2)$), memory usage is drastically reduced for long sequences.

## 3. Code Structure Map

| Component | File Path | Description |
|-----------|-----------|-------------|
| **Token Merging Logic** | `merging/merge.py` | Implements `token_merge_bipartite2d`, protection logic, and sampling. |
| **Attention Layer** | `vggt/layers/attention.py` | Integrates merging/unmerging into the forward pass. |
| **Backbone** | `vggt/models/aggregator.py` | Manages the transformer blocks and passes `global_merging` flags. |
| **Main Model** | `vggt/models/vggt.py` | Top-level entry point; defines configuration parameters. |
| **Heads** | `vggt/heads/` | `TrackHead`, `DPTHead`, `CameraHead` (standard VGGT components). |

## 4. Configuration Parameters

To enable FastVGGT mode, the following parameters are used (typically in `VGGT` or `Aggregator` initialization):

*   `merging` (int): likely controls the strategy ID or mode (0 = off).
*   `merge_ratio` (float): The ratio of tokens to merge (e.g., `0.9` means merge 90% of mergeable tokens).
*   `global_merging` (bool/list): Flags to enable merging in specific layers (Global Attention layers).
*   `vis_attn_map` (bool): For debugging/visualizing the attention maps (and potentially the effect of merging).

## 5. Summary of Data Flow with FastVGGT

1.  **Input**: $[B, S, 3, H, W]$ images.
2.  **Patch Embed**: Convert to tokens $[B, N, C]$.
3.  **Aggregator Blocks**:
    *   **Frame Attention Blocks**: Standard processing (Intra-frame).
    *   **Global Attention Blocks** (with FastVGGT):
        *   **Partition**: Identify Ref, Salient, Src, Dst tokens.
        *   **Merge**: $N \xrightarrow{\text{merge}} N_{reduced}$.
        *   **Global Attention**: Efficient computation on $N_{reduced}$.
        *   **Unmerge**: $N_{reduced} \xrightarrow{\text{restore}} N$.
4.  **Heads**: Process the full-resolution (unmerged) feature maps for dense predictions.
