#!/usr/bin/env python3
"""Debug script to test surface detection"""

import sys
sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/backend')

from pipeline import NavigationPipeline
import cv2
import numpy as np

# Initialize pipeline
pipeline = NavigationPipeline()

# Load a test image (you can replace with an actual image path)
# For now, let's assume there's a test image in the current directory
test_image_path = "test_image.jpg"

try:
    img = cv2.imread(test_image_path)
    if img is None:
        print("Test image not found. Let's test with a sample detection first.")
        # Try a different test image
        import glob
        images = glob.glob("*.jpg") + glob.glob("*.png")
        if images:
            test_image_path = images[0]
            img = cv2.imread(test_image_path)
            print(f"Using image: {test_image_path}")
        else:
            print("No test images found in current directory")
            sys.exit(1)
    
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_np.shape[:2]
    
    print(f"\n=== TESTING SURFACE DETECTION ===")
    print(f"Image size: {w}x{h}")
    
    # Test 1: Detect target object (phone)
    print("\n--- Test 1: Detecting TARGET (phone) ---")
    target = "phone"
    boxes, logits, phrases = pipeline.predict_with_cpu_fallback(
        model=pipeline.grounding_model,
        image=img_np,
        caption=target,
        box_threshold=0.2,
        text_threshold=0.15
    )
    
    if len(boxes) > 0:
        print(f"✓ Found {len(boxes)} instance(s) of '{target}'")
        for i, box in enumerate(boxes):
            box_scaled = box * np.array([w, h, w, h])
            cx, cy, bw, bh = box_scaled
            x1, y1, x2, y2 = int(cx-bw/2), int(cy-bh/2), int(cx+bw/2), int(cy+bh/2)
            conf = float(logits[i].item()) if logits is not None else 0.5
            print(f"  [{i}] Confidence: {conf:.2f}, Box: ({x1},{y1})-({x2},{y2}), Size: {int(bw)}x{int(bh)}")
            target_bbox = (x1, y1, x2, y2)
    else:
        print(f"✗ No instances of '{target}' found")
        sys.exit(1)
    
    # Test 2: Detect individual surfaces
    print("\n--- Test 2: Detecting individual SURFACES ---")
    test_surfaces = ['books', 'table', 'bed', 'floor', 'stack']
    
    for surface in test_surfaces:
        boxes, logits, phrases = pipeline.predict_with_cpu_fallback(
            model=pipeline.grounding_model,
            image=img_np,
            caption=surface,
            box_threshold=0.2,
            text_threshold=0.15
        )
        
        if len(boxes) > 0:
            print(f"✓ Found {len(boxes)} instance(s) of '{surface}'")
            for i, box in enumerate(boxes):
                box_scaled = box * np.array([w, h, w, h])
                cx, cy, bw, bh = box_scaled
                x1, y1, x2, y2 = int(cx-bw/2), int(cy-bh/2), int(cx+bw/2), int(cy+bh/2)
                conf = float(logits[i].item()) if logits is not None else 0.5
                print(f"  [{i}] Confidence: {conf:.2f}, Box: ({x1},{y1})-({x2},{y2}), Size: {int(bw)}x{int(bh)}")
        else:
            print(f"✗ No instances of '{surface}' found")
    
    # Test 3: Use the full detect_spatial_relationships function
    print("\n--- Test 3: Using detect_spatial_relationships() ---")
    surfaces = pipeline.detect_spatial_relationships(img_np, target_bbox, target)
    
    if surfaces:
        print(f"✓ Detected {len(surfaces)} surface(s):")
        for s in surfaces:
            print(f"  - {s['surface']} (confidence: {s['confidence']:.2f})")
    else:
        print(f"✗ No surfaces detected using the full function")
        
        # Debug: Let's manually check the 4-factor logic
        print("\n--- Debug: Manual 4-factor check for 'books' ---")
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        target_cx = (target_x1 + target_x2) / 2
        target_height = target_y2 - target_y1
        target_width = target_x2 - target_x1
        print(f"Target bounding box: ({target_x1}, {target_y1})-({target_x2}, {target_y2})")
        print(f"Target center: ({target_cx:.0f}, {target_y1:.0f})")
        print(f"Target size: {target_width:.0f}x{target_height:.0f}")
        
        # Detect books
        boxes, logits, _ = pipeline.predict_with_cpu_fallback(
            model=pipeline.grounding_model,
            image=img_np,
            caption='books',
            box_threshold=0.2,
            text_threshold=0.15
        )
        
        if len(boxes) > 0:
            print(f"\nFound books, checking 4 factors:")
            for i, box in enumerate(boxes):
                box_scaled = box * np.array([w, h, w, h])
                surf_cx, surf_cy, surf_bw, surf_bh = box_scaled
                surf_x1 = int(surf_cx - surf_bw/2)
                surf_y1 = int(surf_cy - surf_bh/2)
                surf_x2 = int(surf_cx + surf_bw/2)
                surf_y2 = int(surf_cy + surf_bh/2)
                
                print(f"\n  Books instance {i}:")
                print(f"    Books box: ({surf_x1}, {surf_y1})-({surf_x2}, {surf_y2})")
                
                # 4-FACTOR CHECK
                above_surface = target_y2 <= surf_y2
                print(f"    1. Above surface (target_y2 {target_y2} <= surf_y2 {surf_y2}): {above_surface}")
                
                horizontal_overlap = (surf_x1 - 0.3*surf_bw <= target_cx <= surf_x2 + 0.3*surf_bw)
                print(f"    2. Horizontal overlap ({surf_x1 - 0.3*surf_bw:.0f} <= {target_cx:.0f} <= {surf_x2 + 0.3*surf_bw:.0f}): {horizontal_overlap}")
                
                vertical_proximity = target_y2 <= surf_y2 + max(int(0.15*surf_bh), 30)
                print(f"    3. Vertical proximity (target_y2 {target_y2} <= {surf_y2 + max(int(0.15*surf_bh), 30)}): {vertical_proximity}")
                
                size_reasonable = (target_height < 0.7*surf_bh and target_width < 0.7*surf_bw)
                print(f"    4. Size reasonable ({target_height:.0f} < {0.7*surf_bh:.0f} AND {target_width:.0f} < {0.7*surf_bw:.0f}): {size_reasonable}")
                
                all_pass = above_surface and horizontal_overlap and vertical_proximity and size_reasonable
                print(f"    ✓ All 4 factors pass: {all_pass}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
