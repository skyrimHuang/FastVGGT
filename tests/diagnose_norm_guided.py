"""
Diagnostic Analysis: Why is Norm-guided Performing Poorly?
============================================================

Investigating potential issues in the Norm-guided token splitting implementation.
"""

import torch
import sys
sys.path.insert(0, '/home/hba/Documents/FastVGGT')

from merging.merge import token_merge_bipartite2d


def test_norm_guided_partitioning():
    """Analyze the token partitioning in norm-guided method."""
    print("\n" + "="*100)
    print("TEST 1: Token Partitioning Analysis")
    print("="*100)
    
    # Simulate multi-frame input (like ScanNet: 3 images, 37x37 patches + 5 cls tokens each)
    B = 1
    w, h = 37, 37
    tokens_per_img = w * h + 5  # 1374 tokens per image
    num_imgs = 3
    N = tokens_per_img * num_imgs  # 4122 tokens total
    C = 64
    
    # Create dummy metric (random features)
    metric = torch.randn(B, N, C)
    
    # Test norm-guided
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        metric,
        w=w, h=h, sx=1, sy=1, r=50,
        enable_protection=True,
        use_norm_guided=True,
        use_variance=False
    )
    
    print(f"\n📊 Input Shape: {metric.shape}")
    print(f"   - Tokens per image: {tokens_per_img}")
    print(f"   - Number of images: {num_imgs}")
    print(f"   - Total tokens: {N}")
    
    # Analyze token distribution
    # Compute norms for each token
    token_norms = metric[0].norm(dim=-1)  # [N]
    
    print(f"\n📈 L2 Norm Statistics:")
    print(f"   - Mean norm: {token_norms.mean():.4f}")
    print(f"   - Std norm: {token_norms.std():.4f}")
    print(f"   - Min norm: {token_norms.min():.4f}")
    print(f"   - Max norm: {token_norms.max():.4f}")
    
    # Frame-wise analysis
    for img_idx in range(num_imgs):
        start = img_idx * tokens_per_img
        end = start + tokens_per_img
        frame_norms = token_norms[start:end]
        print(f"\n   Frame {img_idx} (tokens {start}-{end-1}):")
        print(f"      Mean norm: {frame_norms.mean():.4f}, Std: {frame_norms.std():.4f}")
        print(f"      High norm (top 10%): {torch.topk(frame_norms, int(0.1*len(frame_norms))).values.mean():.4f}")
        print(f"      Low norm (bot 10%): {torch.topk(frame_norms, int(0.1*len(frame_norms)), largest=False).values.mean():.4f}")


def test_partition_ratios():
    """Compare different partition ratios."""
    print("\n" + "="*100)
    print("TEST 2: Impact of Partition Ratios")
    print("="*100)
    
    # Simulate remaining tokens (after frame 0)
    tokens_per_img = 1374
    num_remaining_imgs = 2
    N_remaining = tokens_per_img * num_remaining_imgs  # 2748 tokens
    
    ratios = [
        ("Current (Norm-guided)", [0.10, 0.10, 0.80]),
        ("Conservative", [0.10, 0.20, 0.70]),
        ("Moderate", [0.10, 0.30, 0.60]),
        ("Variance Top-K", [0.10, 0.40, 0.50]),
        ("Aggressive", [0.05, 0.10, 0.85]),
    ]
    
    print(f"\n📊 Analysis for {N_remaining} remaining tokens (2 images × 1374)")
    print(f"\n{'Strategy':<25} {'Protected':<15} {'Dst (Anchors)':<18} {'Src (Merge)':<15} {'Compression':<12}")
    print("-" * 95)
    
    for name, (p_ratio, d_ratio, s_ratio) in ratios:
        num_protected = int(N_remaining * p_ratio)
        num_dst = int(N_remaining * d_ratio)
        num_src = int(N_remaining * s_ratio)
        
        # Compression ratio: how many src per dst
        compression = num_src / num_dst if num_dst > 0 else 0
        
        print(f"{name:<25} {num_protected:>5} ({p_ratio:.1%}){' '*4} {num_dst:>5} ({d_ratio:.1%}){' '*4} {num_src:>5} ({s_ratio:.1%}){' '*4} {compression:>5.1f}x")
    
    print(f"\n⚠️  KEY INSIGHT:")
    print(f"   - Current norm-guided uses 8x compression (2748 src / 274 dst)")
    print(f"   - This means each dst token receives ~8 merged src tokens")
    print(f"   - High compression → information loss for geometry")


def test_norm_distribution():
    """Check if norm values show meaningful separation."""
    print("\n" + "="*100)
    print("TEST 3: L2 Norm Distribution Analysis")
    print("="*100)
    
    # Create synthetic features with different properties
    torch.manual_seed(42)
    
    # High-importance tokens (e.g., corners, edges)
    high_importance = torch.randn(100, 64) * 2.0 + 1.0  # Higher norm
    
    # Medium-importance tokens
    medium_importance = torch.randn(200, 64) * 1.0  # Medium norm
    
    # Low-importance tokens (redundant regions)
    low_importance = torch.randn(48, 64) * 0.5  # Lower norm
    
    # Combine
    all_features = torch.cat([high_importance, medium_importance, low_importance], dim=0)
    
    # Compute norms
    all_norms = all_features.norm(dim=-1)
    
    print(f"\nSynthetic Feature Norms:")
    print(f"   High importance:   {high_importance.norm(dim=-1).mean():.4f} ± {high_importance.norm(dim=-1).std():.4f}")
    print(f"   Medium importance: {medium_importance.norm(dim=-1).mean():.4f} ± {medium_importance.norm(dim=-1).std():.4f}")
    print(f"   Low importance:    {low_importance.norm(dim=-1).mean():.4f} ± {low_importance.norm(dim=-1).std():.4f}")
    
    # Check if top-10% sorting correctly identifies high-importance tokens
    sorted_indices = torch.argsort(all_norms, descending=True)
    top_10_percent = sorted_indices[:int(0.1 * len(all_norms))]
    
    # Count how many top-10% tokens belong to high_importance
    high_importance_in_top = (top_10_percent < 100).sum().item()
    total_high_importance = 100
    
    print(f"\n✓ Top-10% tokens selection:")
    print(f"   - High-importance tokens in top-10%: {high_importance_in_top}/{total_high_importance} ({100*high_importance_in_top/total_high_importance:.1f}%)")
    print(f"   - This shows norm-based sorting DOES work correctly for importance")
    
    # But check the actual token count
    threshold_10_percent = all_norms[sorted_indices[int(0.1 * len(all_norms))]]
    threshold_30_percent = all_norms[sorted_indices[int(0.3 * len(all_norms))]]
    
    print(f"\n   - Norm threshold for top-10%: {threshold_10_percent:.4f}")
    print(f"   - Norm threshold for top-30%: {threshold_30_percent:.4f}")
    print(f"\n   ⚠️  WARNING: With only 10% dst tokens, many important features")
    print(f"      might get filtered into the remaining 80% (src) category")


def analyze_variance_vs_norm():
    """Compare variance and norm as importance metrics."""
    print("\n" + "="*100)
    print("TEST 4: Variance vs L2 Norm as Importance Metrics")
    print("="*100)
    
    # Create test features
    torch.manual_seed(42)
    N = 1000
    C = 64
    
    features = torch.randn(N, C)
    
    # Compute both metrics
    norms = features.norm(dim=-1)
    variances = features.var(dim=-1)
    
    # Rank tokens by each metric
    norm_ranks = torch.argsort(norms, descending=True)
    var_ranks = torch.argsort(variances, descending=True)
    
    # Check overlap in top-10% selections
    top_10_norm = set(norm_ranks[:100].tolist())
    top_10_var = set(var_ranks[:100].tolist())
    overlap = len(top_10_norm & top_10_var)
    
    print(f"\nComparison of Importance Metrics:")
    print(f"   - L2 Norm: Magnitude of feature vector")
    print(f"   - Variance: Spread of feature values")
    
    print(f"\n   Top-10% selection agreement: {overlap}/100 tokens match ({overlap}%)")
    print(f"   - High overlap (>70%) → metrics are correlated")
    print(f"   - Low overlap (<30%) → metrics capture different aspects")
    
    if overlap < 50:
        print(f"\n   ⚠️  Variance and Norm select different tokens!")
        print(f"   This explains why Variance performs differently from Norm-guided")
    else:
        print(f"\n   ✓ Variance and Norm are well-correlated")


def suggest_fixes():
    """Suggest potential fixes for norm-guided method."""
    print("\n" + "="*100)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*100)
    
    print("""
🔍 ROOT CAUSE ANALYSIS:

The 0.1/0.1/0.8 partition ratio (current Norm-guided) leads to:
  1. Only 10% of tokens serve as merge targets (dst)
  2. 80% of tokens must be merged to these 10% targets
  3. Each target receives ~8 merged source tokens on average
  4. High compression → severe information loss for 3D geometry

Comparison with Variance Top-K (0.1/0.4/0.5):
  - 40% of tokens serve as merge targets (4x more)
  - 50% of tokens must be merged (lower compression)
  - Each target receives ~1.25 merged source tokens
  - Better geometry preservation

═══════════════════════════════════════════════════════════════════════════

🛠️  RECOMMENDED FIXES (Priority Order):

1. ⭐⭐⭐ IMMEDIATE: Adjust Norm-guided partition ratio
   Current spec: 0.1/0.1/0.8 (from thesis section 3.2.2)
   Suggested: Try 0.1/0.3/0.6 or 0.1/0.25/0.65
   
   Rationale: Reduce compression ratio from 8x → 3x or 2.6x
   This preserves geometry while maintaining method difference from Grid-based
   
   Test script: python tests/optimize_norm_partition_ratio.py

2. ⭐⭐ ALTERNATIVE: Verify ratio in thesis spec
   Check if 0.1/0.1/0.8 truly appears in section 3.2.2
   OR if it's a transcription error (should be 0.1/0.3/0.8 or 0.1/0.2/0.7?)
   
   Action: Re-read thesis context and FASTVGGT_CONTEXT.md

3. ⭐⭐ OPTION: Investigate norm computation
   Is L2 norm computed BEFORE or AFTER cosine normalization?
   Current: norm = features.norm(dim=-1) [BEFORE cosine norm]
   Alternative: Compute from cosine-normalized features [AFTER]
   
   Difference: Could significantly change token ranking

4. ⭐ OPTION: Scale norm by spatial distance
   High-norm tokens near image edges might be less useful for geometry
   Suggestion: weight_norm = norm * spatial_prior
   This could improve feature selection

═══════════════════════════════════════════════════════════════════════════

📊 NEXT STEPS:

Option A - Quick Fix (1-2 hours):
  1. Adjust ratio to 0.1/0.3/0.6 in merge.py line 127
  2. Re-run tests/run_token_split_ablation.py
  3. Compare results

Option B - Comprehensive Analysis (4-6 hours):
  1. Create sensitivity analysis script
  2. Test multiple ratios: [0.1,0.2,0.3,0.4,0.5,0.6]
  3. Plot accuracy vs compression ratio
  4. Find optimal balance

Option C - Investigate Spec (2-3 hours):
  1. Review thesis section 3.2.2 context
  2. Check if 0.1/0.1/0.8 is indeed the intended spec
  3. Verify against original paper/algorithm design

═══════════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    test_norm_guided_partitioning()
    test_partition_ratios()
    test_norm_distribution()
    analyze_variance_vs_norm()
    suggest_fixes()
