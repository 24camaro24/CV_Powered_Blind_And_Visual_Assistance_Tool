#!/usr/bin/env python3
"""Quick test of surface detection fix"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/backend')

from pipeline import NavigationPipeline
import cv2
import glob

print("\n" + "="*70)
print("TESTING SURFACE DETECTION FIX")
print("="*70 + "\n")

try:
    print("[1] Initializing pipeline...")
    pipeline = NavigationPipeline()
    print("✓ Pipeline ready\n")
    
    # Find test image
    test_images = glob.glob("/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/**/*.jpg", recursive=True)
    
    if not test_images:
        print("❌ No test images found")
    else:
        print(f"[2] Found {len(test_images)} test image(s)")
        test_img = test_images[0]
        print(f"    Using: {os.path.basename(test_img)}\n")
        
        # Process the image
        print("[3] Processing image...")
        target = "bottle"  # or "water bottle" - try to detect
        
        result = pipeline.process_image(test_img, target)
        
        if result['success']:
            print(f"✓ Processing successful\n")
            
            print(f"Target detected: {result.get('target')}")
            print(f"Distance: {result.get('distance_meters'):.2f}m")
            print(f"Confidence: {result.get('confidence', 0)*100:.1f}%\n")
            
            # Check surfaces
            surfaces = result.get('surfaces', [])
            print(f"SURFACES DETECTED: {len(surfaces)}")
            
            if surfaces:
                print("✅ Surfaces found:")
                for surf in surfaces:
                    print(f"   - {surf['surface']} (conf: {surf['confidence']:.2f})")
            else:
                print("⚠ No surfaces detected (this is the issue we're fixing)")
            
            # Now test generate_instruction (what app.py does)
            print(f"\n[4] Testing generate_instruction with surfaces...")
            instruction = pipeline.generate_instruction(
                result['target'],
                result['steps'],
                result['angle'],
                result['distance_meters'],
                result.get('confidence', 0.85),
                result.get('depth', 0),
                surfaces=surfaces
            )
            
            voice = instruction.get('conversational', '')
            
            print(f"\nVoice Output (from generate_instruction):")
            print(f'"{voice}"')
            
            # Check if surfaces are in voice
            if surfaces:
                has_surface_context = any(s['surface'].lower() in voice.lower() for s in surfaces)
                if has_surface_context:
                    print(f"\n✅ Surface context IS in voice output!")
                else:
                    print(f"\n❌ Surface context NOT in voice output")
                    print(f"   Expected to find: {', '.join([s['surface'] for s in surfaces[:3]])}")
            
        else:
            print(f"❌ Processing failed: {result.get('error')}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
