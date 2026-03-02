"""
Full Implementation Chain Debugging
====================================

Trace the exact execution path from config to actual merging.
"""

import torch
import sys
sys.path.insert(0, '/home/hba/Documents/FastVGGT')

from merging.merge import token_merge_bipartite2d


def debug_merging_execution():
    """Debug the merging execution path with detailed logging."""
    
    print("\n" + "="*100)
    print("DEBUGGING: Token Merge Execution Path")
    print("="*100)
    
    # Setup
    B = 1
    w, h = 37, 37
    tokens_per_img = w * h + 5  # 1374
    num_imgs = 3
    N = tokens_per_img * num_imgs  # 4122
    C = 64
    
    metric = torch.randn(B, N, C)
    r = int(N * 0.7)  # merge ratio
    
    print(f"\n📊 INPUT CONFIGURATION:")
    print(f"   Batch: {B}, Total tokens: {N}, Feature dim: {C}")
    print(f"   Grid: {w}x{h}, Images: {num_imgs}")
    print(f"   Merge ratio: {r/N:.1%} (r={r})")
    
    # Test 1: Grid-based (default)
    print(f"\n{'='*100}")
    print("TEST 1: Grid-based (default, enable_protection=False)")
    print(f"{'='*100}")
    
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        metric.clone(),
        w=w, h=h, sx=1, sy=1, r=r,
        enable_protection=False,  # NO PROTECTION
        use_norm_guided=False,
        use_variance=False
    )
    print(f"   ✓ Merge function created")
    print(f"   - enable_protection: False")
    print(f"   - use_norm_guided: False")
    print(f"   - use_variance: False")
    
    # Test merge/unmerge
    dummy_x = torch.randn(B, N, C)
    merged = merge_fn(dummy_x)
    print(f"   - Input shape: {dummy_x.shape}")
    print(f"   - Merged shape: {merged.shape}")
    
    # Test 2: Grid-based WITH protection
    print(f"\n{'='*100}")
    print("TEST 2: Grid-based WITH protection (enable_protection=True)")
    print(f"{'='*100}")
    
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        metric.clone(),
        w=w, h=h, sx=1, sy=1, r=r,
        enable_protection=True,  # ENABLE PROTECTION
        use_norm_guided=False,
        use_variance=False
    )
    print(f"   ✓ Merge function created")
    print(f"   - enable_protection: True")
    print(f"   - use_norm_guided: False")
    print(f"   - use_variance: False")
    
    dummy_x = torch.randn(B, N, C)
    merged = merge_fn(dummy_x)
    print(f"   - Input shape: {dummy_x.shape}")
    print(f"   - Merged shape: {merged.shape}")
    
    # Test 3: Norm-guided (PRIMARY TEST)
    print(f"\n{'='*100}")
    print("TEST 3: NORM-GUIDED (use_norm_guided=True, enable_protection=True)")
    print(f"{'='*100}")
    
    metric_clone = metric.clone()
    
    # Verify this is normalization-sensitive
    print(f"\n   Computing token norms (BEFORE applying norm-guided):")
    token_norms = metric_clone[0].norm(dim=-1)  # [N]
    print(f"   - Min norm: {token_norms.min():.4f}")
    print(f"   - Max norm: {token_norms.max():.4f}")
    print(f"   - Mean norm: {token_norms.mean():.4f}")
    
    # Frame-wise norm statistics
    for img_idx in range(num_imgs):
        start = img_idx * tokens_per_img
        end = start + tokens_per_img
        frame_norms = token_norms[start:end]
        sorted_frame_norms = torch.sort(frame_norms, descending=True)
        print(f"   - Frame {img_idx}: top-10% mean norm = {sorted_frame_norms.values[:int(0.1*len(frame_norms))].mean():.4f}")
    
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        metric_clone,
        w=w, h=h, sx=1, sy=1, r=r,
        enable_protection=True,  # CRITICAL
        use_norm_guided=True,     # CRITICAL
        use_variance=False
    )
    print(f"\n   ✓ Merge function created")
    print(f"   - enable_protection: True ← MUST BE TRUE for norm-guided to execute")
    print(f"   - use_norm_guided: True ← MUST BE TRUE to select norm-guided branch")
    print(f"   - use_variance: False")
    
    dummy_x = torch.randn(B, N, C)
    merged = merge_fn(dummy_x)
    print(f"\n   - Input shape: {dummy_x.shape}")
    print(f"   - Merged shape: {merged.shape}")
    print(f"   - Expected reduction: ✓ Tokens reduced" if merged.shape[1] < N else "   - WARNING: No reduction detected")
    
    # Test 4: Variance Top-K
    print(f"\n{'='*100}")
    print("TEST 4: VARIANCE TOP-K (use_variance=True, enable_protection=True)")
    print(f"{'='*100}")
    
    metric_clone = metric.clone()
    print(f"\n   Computing token variances (BEFORE applying variance-guided):")
    token_vars = metric_clone[0].var(dim=-1, unbiased=False)  # [N]
    print(f"   - Min variance: {token_vars.min():.4f}")
    print(f"   - Max variance: {token_vars.max():.4f}")
    print(f"   - Mean variance: {token_vars.mean():.4f}")
    
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        metric_clone,
        w=w, h=h, sx=1, sy=1, r=r,
        enable_protection=True,   # CRITICAL
        use_norm_guided=False,
        use_variance=True         # CRITICAL
    )
    print(f"\n   ✓ Merge function created")
    print(f"   - enable_protection: True")
    print(f"   - use_norm_guided: False")
    print(f"   - use_variance: True ← MUST BE TRUE to select variance branch")
    
    dummy_x = torch.randn(B, N, C)
    merged = merge_fn(dummy_x)
    print(f"\n   - Input shape: {dummy_x.shape}")
    print(f"   - Merged shape: {merged.shape}")
    print(f"   - Expected reduction: ✓ Tokens reduced" if merged.shape[1] < N else "   - WARNING: No reduction detected")


def check_enable_protection_in_actual_flow():
    """
    Check if enable_protection is actually True in the real eval flow.
    """
    
    print(f"\n\n{'='*100}")
    print("CRITICAL CHECK: enable_protection Flag Status")
    print(f"{'='*100}")
    
    print(f"""
⚠️  POTENTIAL ISSUE IDENTIFIED:

In vggt/layers/attention.py line ~188, the merging call is:

    m, u = token_merge_bipartite2d(
        x,
        self.patch_width,
        self.patch_height,
        2,
        2,
        r,
        False,
        generator,
        enable_protection=True,              ← IS THIS ALWAYS TRUE?
        use_norm_guided=self.use_norm_guided,
        use_variance=self.use_variance,
    )

✓ Good news: enable_protection IS hardcoded to True
✓ use_norm_guided IS correctly passed from self

However, let's verify protection is actually working in merge.py:

In merge.py, the norm-guided branch execution requires:
    1. use_norm_guided == True         ✓ Passed correctly
    2. enable_protection == True       ✓ Hardcoded to True
    3. Both conditions met (line ~99): "if use_norm_guided and enable_protection:"

So theoretically, if use_norm_guided=True is passed, norm-guided SHOULD execute.

But the test results show ALL ratios return IDENTICAL metrics.
This suggests the norm-guided branch is either:
    A) Calculating but not actually affecting the partition (logic error)
    B) Not being executed due to some other condition
    C) Being overridden by default grid-based partition logic

Next: Examine the actual partition logic in merge.py...
    """)


def trace_merge_logic():
    """
    Trace the actual merging logic step by step.
    """
    print(f"\n\n{'='*100}")
    print("TRACING: Actual Partition Logic in merge.py")
    print(f"{'='*100}")
    
    print(f"""
Key observation from merge.py lines 98-210:

1️⃣  Lines 98-107: NORM-GUIDED BRANCH
   if use_norm_guided and enable_protection:
       [Compute L2 norms]
       [Sort by norm descending]
       [Partition: 0.10/0.10/0.80]
       [Set idx_buffer_seq values to -2/-1/0+]

2️⃣  Lines 188-210: GRID-BASED BRANCH
   if not use_norm_guided:
       [Standard grid partition]
       [Set idx_buffer_seq values]

⚠️  CRITICAL ISSUE FOUND:

Line 217 (approx):
    if not use_norm_guided:
        # Mark first image entirely as dst
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

This means:
- If use_norm_guided=True, the norm-guided branch executes
- If use_norm_guided=False, the grid-based branch executes

But there's a subtle issue...

Let me check line 217-241 more carefully for the condition...
    """)


if __name__ == "__main__":
    debug_merging_execution()
    check_enable_protection_in_actual_flow()
    trace_merge_logic()
