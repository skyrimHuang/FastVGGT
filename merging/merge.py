import torch
from typing import Tuple, Callable, Optional, Union


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    use_norm_guided: bool = False,
    norm_protected_ratio: float = 0.10,
    norm_dst_ratio: float = 0.40,
) -> Tuple[Callable, Callable]:
    """
    Divide tokens into source (src) and destination (dst) groups, and merge r tokens from src to dst.
    
    Strategies:
    1. Grid-based split (default): Divide tokens spatially into grid blocks
    2. Norm-Guided Anchoring: Use L2 norm to protect high-importance tokens
    
    Merging uses Top-K strategy: Select top r pairs with highest cosine similarity.

    Args:
     - metric [B, N, C]: Tensor for similarity computation, B=batch size, N=token count, C=feature dimension
     - w: Image width in tokens
     - h: Image height in tokens
     - sx: dst stride in x dimension, must divide w evenly
     - sy: dst stride in y dimension, must divide h evenly
     - r: Number of tokens to remove through merging (Top-K parameter)
     - no_rand: If True, disable randomness (use only top-left token)
     - generator: Random number generator if no_rand is False and not None
     - enable_protection: If True, enable importance protection (protect high-norm tokens)
     - use_norm_guided: If True, use L2 norm-based split instead of grid-based
    - norm_protected_ratio: Norm-guided protected ratio
    - norm_dst_ratio: Norm-guided dst ratio

    Returns:
     - (merge, unmerge): Two functions for merging tokens and restoring pre-merge state
    """
    B, N, _ = metric.shape  # Batch size B, total tokens N
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather

    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    with torch.no_grad():
        # ============ Norm-Guided Anchoring (方案一) ============
        if use_norm_guided and enable_protection:
            # Initialize idx_buffer per batch: [B, N]
            idx_buffer_seq = torch.zeros(B, N, device=metric.device, dtype=torch.int64)

            norm_protected_ratio = float(norm_protected_ratio)
            norm_dst_ratio = float(norm_dst_ratio)
            
            # ============ FRAME 0: All tokens as dst ============
            if num_imgs > 0:
                idx_buffer_seq[:, :tokens_per_img] = -1  # Frame0: all dst (reference frame)
            
            # ============ FRAME 1~N: Per-frame independent partitioning ============
            # Each frame independently sorts/partitions for EACH sample in batch
            if num_imgs > 1:
                src_token_counter = 0
                batch_arange = torch.arange(B, device=metric.device).unsqueeze(1)
                
                for frame_idx in range(1, num_imgs):
                    # Keep cls/register tokens of each non-anchor frame as dst
                    frame_token_start = frame_idx * tokens_per_img
                    frame_cls_end = frame_token_start + 5
                    idx_buffer_seq[:, frame_token_start:frame_cls_end] = -1

                    # Calculate patch token range for this frame (skip 5 cls/pos tokens)
                    frame_patch_start = frame_idx * tokens_per_img + 5
                    frame_patch_end = frame_patch_start + (h * w)
                    
                    # Extract this frame's patch tokens only
                    frame_tokens = metric[:, frame_patch_start:frame_patch_end, :]  # [B, h*w, C]
                    
                    # Compute L2 norm per sample within this frame only
                    frame_norms = frame_tokens.norm(dim=-1)  # [B, h*w]
                    # Per-sample independent sorting in this frame
                    sorted_indices_local = torch.argsort(frame_norms, dim=-1, descending=True)  # [B, h*w]
                    
                    # Three-tier partitioning for this frame only (per-frame independent ratio)
                    n_frame_tokens = sorted_indices_local.shape[1]
                    n_protected = int(n_frame_tokens * norm_protected_ratio)  # e.g., 10%
                    n_dst = int(n_frame_tokens * norm_dst_ratio)              # e.g., 40%
                    # remaining tokens are src (50%)
                    
                    # Extract per-sample indices for this frame
                    protected_local = sorted_indices_local[:, :n_protected]  # [B, n_protected]
                    dst_local = sorted_indices_local[:, n_protected:n_protected + n_dst]  # [B, n_dst]
                    src_local = sorted_indices_local[:, n_protected + n_dst:]  # [B, n_src]
                    
                    # Convert to global indices (per sample)
                    protected_global = frame_patch_start + protected_local
                    dst_global = frame_patch_start + dst_local
                    src_global = frame_patch_start + src_local
                    
                    # Mark in idx_buffer: -2=protected, -1=dst, 0+=src counter
                    if n_protected > 0:
                        idx_buffer_seq[batch_arange, protected_global] = -2
                    if n_dst > 0:
                        idx_buffer_seq[batch_arange, dst_global] = -1

                    n_src = src_local.shape[1]
                    if n_src > 0:
                        src_values = torch.arange(
                            src_token_counter,
                            src_token_counter + n_src,
                            device=metric.device,
                            dtype=torch.int64,
                        ).unsqueeze(0).expand(B, -1)
                        idx_buffer_seq[batch_arange, src_global] = src_values
                        src_token_counter += n_src

                # Build protected indices per sample: [B, num_protected_actual]
                protected_indices_list = [
                    torch.where(idx_buffer_seq[b] == -2)[0] for b in range(B)
                ]
                if protected_indices_list and protected_indices_list[0].numel() > 0:
                    protected_indices = torch.stack(protected_indices_list, dim=0)
                else:
                    protected_indices = torch.empty(B, 0, device=metric.device, dtype=torch.long)
            else:
                # Single frame: all tokens as dst, no protected tokens needed
                protected_indices = torch.empty(B, 0, device=metric.device, dtype=torch.long)
            
        # ============ Original Grid-Based Split (fallback) ============
        elif enable_protection:
            # Sample protected tokens only from non-anchor frames to avoid re-labeling frame-0 dst tokens
            if num_imgs > 1:
                remaining_start = tokens_per_img
                N_remaining = N - tokens_per_img
                num_protected = int(N_remaining * 0.1)
                if num_protected > 0:
                    step = max(1, N_remaining // num_protected)
                    protected_indices = remaining_start + torch.arange(
                        0, N_remaining, step, device=metric.device
                    )[:num_protected]
                else:
                    protected_indices = torch.empty(0, device=metric.device, dtype=torch.long)
            else:
                num_protected = 0
                protected_indices = torch.empty(0, device=metric.device, dtype=torch.long)
            idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        else:
            protected_indices = None
            num_protected = 0
            idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx  # Number of blocks within each image

        # Skip grid-based split if using norm-guided (already done above)
        if not use_norm_guided:
            # Mark first image entirely as dst
            if num_imgs > 0:
                idx_buffer_seq[:tokens_per_img] = -1

            # Process other images - fully vectorized batch operations
            if num_imgs > 1:
                cls_indices = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
                )
                cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
                idx_buffer_seq[cls_indices.flatten()] = -1
                effective_h = min(hsy * sy, h)
                effective_w = min(wsx * sx, w)
                effective_grid_size = effective_h * effective_w

                if no_rand:
                    base_pattern = torch.zeros(
                        effective_grid_size, device=metric.device, dtype=torch.int64
                    )
                    grid_starts = (
                        torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                    )
                    grid_indices = grid_starts[:, None] + torch.arange(
                        effective_grid_size, device=metric.device
                    )
                    idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                        num_imgs - 1
                    )
                else:
                    total_other_imgs = num_imgs - 1
                    all_rand_idx = torch.randint(
                        sy * sx,
                        size=(total_other_imgs, hsy, wsx),
                        device=metric.device,
                        generator=generator,
                    )

                    scatter_src = -torch.ones(
                        total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                    )

                    idx_buffer_batch = torch.zeros(
                        total_other_imgs,
                        hsy,
                        wsx,
                        sy * sx,
                        device=metric.device,
                        dtype=torch.int64,
                    )
                    idx_buffer_batch.scatter_(
                        dim=3,
                        index=all_rand_idx.unsqueeze(-1),
                        src=scatter_src.unsqueeze(-1),
                    )

                    idx_buffer_batch = (
                        idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                        .transpose(2, 3)
                        .reshape(total_other_imgs, hsy * sy, wsx * sx)
                    )

                    # Batch fill to target positions - still needs a small loop here, but operations are greatly reduced
                    for i in range(total_other_imgs):
                        img_idx = i + 1
                        grid_start = img_idx * tokens_per_img + 5
                        flat_view = idx_buffer_batch[
                            i, :effective_h, :effective_w
                        ].flatten()
                        idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                            flat_view
                        )
        
        # 🔧 FIX: Mark protected tokens for grid-based method (consistency with norm-guided)
        if enable_protection and (not use_norm_guided):
            idx_buffer_seq[protected_indices] = -2  # Protected tokens: -2

        # 🔧 FIX: Sort idx_buffer and correctly partition into protected/dst/src
        # Support both legacy [N] and per-batch [B, N] layouts
        if idx_buffer_seq.dim() == 1:
            idx_buffer = idx_buffer_seq.unsqueeze(0).expand(B, -1)
        else:
            idx_buffer = idx_buffer_seq

        rand_idx = idx_buffer.reshape(B, -1, 1).argsort(dim=1)

        # Count each type per batch and ensure shape consistency across batch
        if enable_protection:
            num_protected_each = (idx_buffer == -2).sum(dim=1)
            num_protected_buffer = int(num_protected_each[0].item())
            if not torch.all(num_protected_each == num_protected_buffer):
                raise RuntimeError("Inconsistent protected token counts across batch")
        else:
            num_protected_buffer = 0

        num_dst_each = (idx_buffer == -1).sum(dim=1)
        num_dst_orig = int(num_dst_each[0].item())
        if not torch.all(num_dst_each == num_dst_orig):
            raise RuntimeError("Inconsistent dst token counts across batch")

        num_src_buffer = N - num_protected_buffer - num_dst_orig
        
        # Partition indices: [protected | dst | src]
        # Protected tokens should NOT be in a_idx, they're handled separately via protected_idx
        if enable_protection and num_protected_buffer > 0:
            # Skip protected tokens (first num_protected_buffer after sorting)
            a_idx_orig = rand_idx[:, num_protected_buffer + num_dst_orig:, :]  # Only real src tokens
            b_idx_orig = rand_idx[:, num_protected_buffer:num_protected_buffer + num_dst_orig, :]  # Dst tokens
            # Protected tokens are handled separately via protected_idx
        else:
            # No protection enabled (all tokens treated as src or dst)
            a_idx_orig = rand_idx[:, num_dst_orig:, :]
            b_idx_orig = rand_idx[:, :num_dst_orig, :]
        
        a_idx = a_idx_orig
        b_idx = b_idx_orig

        if enable_protection:
            if protected_indices.dim() == 1:
                protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
            else:
                protected_idx = protected_indices.unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        # Define an internal function to separate src, dst, and protected tokens
        def split(x):
            C = x.shape[-1]

            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        # Compute cosine similarity (normalize first then dot product)
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        # 🔧 FIX: Force merge ALL src tokens to dst (no unmerged src tokens)
        # src tokens have already been separated from protected tokens via split()
        # so no additional filtering is needed
        num_src_actual = a.shape[1]
        
        # For norm-guided or forced complete merging: use ALL src tokens
        if use_norm_guided and enable_protection:
            r = num_src_actual  # Merge ALL src tokens
        else:
            r = min(num_src_actual, r)
        
        if num_src_actual == 0:
            node_max = torch.empty(B, 0, device=a.device, dtype=a.dtype)
            node_idx = torch.empty(B, 0, device=a.device, dtype=torch.long)
        else:
            chunk_size = min(5000, num_src_actual)
            node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
            node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)
            b_transposed = b.transpose(-1, -2)
            node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        
        # ============ Top-K Selection (Force All Src Merging) ============
        # Select top r pairs with highest cosine similarity
        # Protected tokens are already separated, so no need for additional filtering
        # ============ Top-K Selection: Force ALL Src Merging ============
        # 🔧 V3: No top-k filtering - ALL src tokens MUST merge to dst
        # For each src token, find the dst token with highest cosine similarity
        # and FORCE the merge (no unmerged src tokens allowed)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        # Force all src tokens to be merged (no unm_idx)
        r_actual = num_src_actual  # 🔧 ALL src tokens must be merged
        
        # Create indices for forced merging:
        # - src_idx: all src tokens [0, 1, 2, ..., num_src_actual-1]
        # - unm_idx: empty (no unmerged src)
        src_idx = edge_idx[..., :r_actual, :]  # All src in merge order
        unm_idx = torch.empty((B, 0, 1), device=metric.device, dtype=torch.long)  # Empty unmerged

        # Get dst token indices corresponding to each src token to be merged
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual

    # Define merge function to merge selected src tokens to corresponding dst tokens
    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape

        # Extract unmerged src tokens - using actual unm_idx size
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        # ---------------- Extra tensor processing ----------------
        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            # Consistent with main tensor, merge src tokens
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            
            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
            
            # Handle unmerged src tokens (should be empty in force-merge mode)
            if unm_len > 0:
                unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))
                if enable_protection:
                    merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
                else:
                    merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)
            else:
                # All src merged, no unm
                if enable_protection:
                    merged_extra_1 = torch.cat([dst_e, protected_e], dim=1)
                else:
                    merged_extra_1 = dst_e

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            
            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
            
            # Handle unmerged src tokens (should be empty in force-merge mode)
            if unm_len > 0:
                unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))
                if enable_protection:
                    merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
                else:
                    merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)
            else:
                # All src merged, no unm
                if enable_protection:
                    merged_extra_2 = torch.cat([dst_e2, protected_e2], dim=1)
                else:
                    merged_extra_2 = dst_e2

        if enable_protection:
            # When all src are merged: output = [dst_merged, protected]
            # unm should be empty, so skip it
            if unm_len > 0:
                main_result = torch.cat([unm, dst, protected], dim=1)
            else:
                main_result = torch.cat([dst, protected], dim=1)
        else:
            # When all src are merged: output = [dst_merged]
            if unm_len > 0:
                main_result = torch.cat([unm, dst], dim=1)
            else:
                main_result = dst

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        else:
            return main_result

    # Define unmerge function to restore pre-merge state (for decoder)
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        
        # When all src are merged: input order is [dst, protected] (no unm)
        # When partial merge: input order is [unm, dst, protected]
        if unm_len > 0:
            unm = x[..., :unm_len, :]
            dst = x[..., unm_len : unm_len + dst_len, :]
        else:
            # No unmerged src - unm is empty
            unm = torch.empty((x.shape[0], 0, x.shape[2]), device=x.device, dtype=x.dtype)
            dst = x[..., :dst_len, :]

        if enable_protection:
            if unm_len > 0:
                protected = x[
                    ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
                ]
            else:
                protected = x[..., dst_len : dst_len + num_protected_actual, :]
        
        # Restore merged src tokens from dst
        _, _, c = dst.shape  # Use dst shape instead of unm (which might be empty)
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        
        # Restore all tokens to their original positions
        # First: restore dst tokens
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        
        # Second: restore unmerged src tokens (if any)
        if unm_len > 0:
            out.scatter_(
                dim=-2,
                index=gather(
                    a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
                ).expand(B, unm_len, c),
                src=unm,
            )
        
        # Third: restore merged src tokens
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        # Fourth: restore protected tokens (if any)
        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge
