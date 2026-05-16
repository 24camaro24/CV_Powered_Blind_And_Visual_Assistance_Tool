#!/usr/bin/env python3
"""Comprehensive test of improved surface detection"""

import sys
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/backend')

from pipeline import NavigationPipeline
import numpy as np

print("\n" + "="*70)
print("TESTING IMPROVED SURFACE DETECTION")
print("="*70)

try:
    print("\n[1] Initializing NavigationPipeline...")
    pipeline = NavigationPipeline()
    print("✓ Pipeline initialized")
    
    # Create a simple synthetic test to validate surface detection logic
    print("\n[2] Testing surface detection parameters...")
    
    # Simulate what would happen with a phone on books
    print("\n   Scenario: Phone is on top of books")
    print("   Target box: (200, 150) - (250, 180)  [phone]")
    print("   Surface box: (100, 175) - (350, 190) [books]")
    
    # These are the scenarios
    target_bbox = (200, 150, 250, 180)  # phone
    surface_bbox = (100, 175, 350, 190)  # books
    
    t_x1, t_y1, t_x2, t_y2 = target_bbox
    s_x1, s_y1, s_x2, s_y2 = surface_bbox
    
    # Extract dimensions
    t_h = t_y2 - t_y1  # 30
    t_w = t_x2 - t_x1  # 50
    s_h = s_y2 - s_y1  # 15
    s_w = s_x2 - s_x1  # 250
    
    t_cx = (t_x1 + t_x2) / 2  # 225
    t_area = t_h * t_w  # 1500
    s_area = s_h * s_w  # 3750
    
    print(f"\n   Target (phone): bbox=({t_x1}, {t_y1})-({t_x2}, {t_y2}), size={t_w}x{t_h}, cx={t_cx}")
    print(f"   Surface (books): bbox=({s_x1}, {s_y1})-({s_x2}, {s_y2}), size={s_w}x{s_h}")
    
    # Check all 4 factors with NEW logic
    print("\n   Checking NEW spatial factors:")
    
    # 1. Vertical
    above_or_on = t_y2 <= s_y2 + max(int(0.1*s_h), 20)
    print(f"   ✓ Vertical: t_y2({t_y2}) <= s_y2({s_y2}) + max(0.1*{s_h}, 20) = {s_y2 + max(int(0.1*s_h), 20)} ? {above_or_on}")
    
    # 2. Horizontal
    horiz_margin = max(0.4 * s_w, 50)
    horiz_align = (s_x1 - horiz_margin <= t_cx <= s_x2 + horiz_margin)
    print(f"   ✓ Horizontal: {s_x1 - horiz_margin:.0f} <= t_cx({t_cx}) <= {s_x2 + horiz_margin:.0f} ? {horiz_align}")
    
    # 3. Size
    smaller = t_area < 0.8 * s_area
    print(f"   ✓ Size: t_area({t_area}) < 0.8 * s_area({s_area:.0f}) ? {smaller}")
    
    all_factors = above_or_on and horiz_align and smaller
    print(f"\n   ✓ Result: All factors pass? {all_factors}")
    
    if all_factors:
        print("\n   ✅ SUCCESS: New algorithm correctly identifies phone is on books!")
    else:
        print("\n   ❌ FAILED: New algorithm still has issues")
        if not above_or_on:
            print(f"      - Vertical check failed: {t_y2} > {s_y2 + max(int(0.1*s_h), 20)}")
        if not horiz_align:
            print(f"      - Horizontal check failed: {t_cx} not in range [{s_x1 - horiz_margin:.0f}, {s_x2 + horiz_margin:.0f}]")
        if not smaller:
            print(f"      - Size check failed: {t_area} >= {0.8 * s_area:.0f}")
    
    # Test with actual image if available
    print("\n[3] Looking for test image...")
    import glob
    test_images = glob.glob("/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/**/*.jpg", recursive=True)
    test_images += glob.glob("/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/**/*.png", recursive=True)
    
    if test_images:
        print(f"✓ Found {len(test_images)} test image(s)")
        # Use the first available image
        test_img_path = test_images[0]
        print(f"  Using: {os.path.basename(test_img_path)}")
        
        import cv2
        img = cv2.imread(test_img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] range
            if img_rgb.dtype == np.uint8:
                img_rgb = img_rgb.astype(np.float32) / 255.0
            print(f"  Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}, dtype: {img_rgb.dtype}, range: [{img_rgb.min():.2f}, {img_rgb.max():.2f}]")
            
            # Try to detect phone
            print("\n[4] Testing with actual image...")
            print("  Detecting 'phone'...")
            boxes, logits, _ = pipeline.predict_with_cpu_fallback(
                model=pipeline.grounding_model,
                image=img_rgb,
                caption='phone',
                box_threshold=0.2,
                text_threshold=0.15
            )
            
            if len(boxes) > 0:
                print(f"  ✓ Found phone ({len(boxes)} instance)")
                box = boxes[0] * np.array([img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[0]])
                cx, cy, bw, bh = box
                x1, y1, x2, y2 = int(cx-bw/2), int(cy-bh/2), int(cx+bw/2), int(cy+bh/2)
                target_bbox = (x1, y1, x2, y2)
                
                print(f"    Box: ({x1}, {y1})-({x2}, {y2})")
                
                # Now test surface detection
                print("\n  Testing surface detection on this image...")
                surfaces = pipeline.detect_spatial_relationships(img_rgb, target_bbox, 'phone')
                
                if surfaces:
                    print(f"  ✅ Detected {len(surfaces)} surface(s):")
                    for surf in surfaces:
                        print(f"     - {surf['surface']} (confidence: {surf['confidence']:.2f})")
                else:
                    print(f"  ❌ No surfaces detected")
            else:
                print(f"  ⚠ Phone not found in test image")
        else:
            print(f"  ⚠ Could not read test image")
    else:
        print("⚠ No test images found - skipping real image test")
    
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
