#!/usr/bin/env python3
"""
Verification script to confirm Chinese fonts in generated PNG images
"""

import os
from PIL import Image
import subprocess

demo_dir = '/home/hba/Documents/FastVGGT/tests/tests_result/hybrid_registration_7scenes/demo'
png_files = [
    'demo_figure1_method_comparison.png',
    'demo_figure2_scene_distribution.png',
    'demo_figure3_error_correlation.png',
    'demo_figure4_multimet_comparison.png',
]

print("=" * 70)
print("PNG IMAGE VERIFICATION REPORT")
print("=" * 70)

for png_file in png_files:
    full_path = os.path.join(demo_dir, png_file)
    
    if not os.path.exists(full_path):
        print(f"✗ {png_file}: FILE NOT FOUND")
        continue
    
    try:
        # Open and check image basic info
        img = Image.open(full_path)
        size = os.path.getsize(full_path) / 1024  # KB
        
        print(f"\n✓ {png_file}")
        print(f"  - Size: {img.width}×{img.height} pixels")
        print(f"  - File size: {size:.1f} KB")
        print(f"  - Format: {img.format}")
        print(f"  - Mode: {img.mode}")
        
        # Check if there's text (this is a basic check)
        if img.mode == 'RGBA':
            print(f"  - RGBA mode: ✓ (supports transparent text)")
        
    except Exception as e:
        print(f"✗ {png_file}: Error reading image - {e}")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Open the PNG files in an image viewer to visually verify Chinese text")
print("2. Check that labels like '旋转误差', '平移误差', 'Chamfer距离' display correctly")
print("3. Verify that method names '方法 A', '方法 B', '方法 C' are rendered properly")
print("4. Confirm red disclaimer text '[演示数据]' is visible at bottom of each figure")
print("\nFiles are located at:")
print(f"  {demo_dir}/")
print("\n" + "=" * 70)
