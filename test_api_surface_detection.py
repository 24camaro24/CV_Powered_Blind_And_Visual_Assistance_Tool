#!/usr/bin/env python3
"""Test the improved surface detection via the API"""

import requests
import json
from pathlib import Path
import base64

print("\n" + "="*70)
print("TESTING IMPROVED SURFACE DETECTION VIA API")
print("="*70)

# First, start the server in background if not already running
import subprocess
import time
import os

server_process = None
try:
    # Check if server is already running
    response = requests.get("http://localhost:5001/", timeout=2)
    print("\n✓ Server already running at localhost:5001")
except:
    print("\n[Attempting to start server...]")
    # Start server in background
    os.chdir('/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code')
    server_process = subprocess.Popen(['python3', 'backend/app.py'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
    print("Waiting for server to start...")
    time.sleep(5)

try:
    # Find a test image
    import glob
    test_images = glob.glob("/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/**/*.jpg", recursive=True)
    test_images += glob.glob("/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/**/*.png", recursive=True)
    
    if not test_images:
        print("❌ No test images found")
    else:
        test_image = test_images[0]
        print(f"\n✓ Using test image: {Path(test_image).name}")
        
        # Test 1: Process image with target detection
        print("\n[Test 1] Processing image to detect 'phone'...")
        with open(test_image, 'rb') as f:
            files = {'image': f}
            data = {'target': 'phone'}
            response = requests.post('http://localhost:5001/api/process', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Image processed successfully")
            
            if result.get('success'):
                print(f"\n  Target: {result.get('target')}")
                print(f"  Steps: {result.get('steps')}")
                print(f"  Angle: {result.get('angle')}°")
                print(f"  Distance: {result.get('distance_meters')}m")
                print(f"  Confidence: {result.get('confidence', 0)*100:.1f}%")
                
                # Check if surfaces were detected
                surfaces = result.get('surfaces', [])
                print(f"\n  SURFACES DETECTED: {len(surfaces)}")
                if surfaces:
                    print(f"  ✅ Surfaces found:")
                    for surf in surfaces:
                        print(f"     - {surf['surface']} (confidence: {surf['confidence']:.2f})")
                else:
                    print(f"  ⚠ No surfaces detected")
                
                # Check voice guidance
                nav_guidance = result.get('navigation_guidance', {})
                voice_text = nav_guidance.get('conversational_text', '')
                print(f"\n  VOICE OUTPUT:")
                print(f"  '{voice_text}'")
                
                # Check if surface context is in the voice
                if surfaces and any(s['surface'].lower() in voice_text.lower() for s in surfaces):
                    print(f"\n  ✅ Surface context IS present in voice output!")
                elif surfaces:
                    print(f"\n  ❌ Surface context NOT present in voice output (surfaces detected but not mentioned)")
                else:
                    print(f"\n  ⚠ Can't verify - no surfaces were detected")
            else:
                print(f"❌ Detection failed: {result.get('error')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(response.text)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if server_process:
        print("\n[Stopping server...]")
        server_process.terminate()
        server_process.wait(timeout=5)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
