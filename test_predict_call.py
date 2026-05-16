#!/usr/bin/env python3
"""Test GroundingDINO predict() to see if CUDA error occurs"""

import signal
import sys
import os
from PIL import Image
from pathlib import Path

os.chdir('/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/quick_depth_estimation_demo_code')
sys.path.insert(0, 'backend')

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

try:
    print('[1] Importing pipeline...')
    from pipeline import NavigationPipeline
    signal.alarm(60)  # 60s timeout for initialization
    
    print('[2] Creating pipeline...')
    pipeline = NavigationPipeline()
    signal.alarm(0)  # Cancel alarm after init
    
    print('[3] Creating test image...')
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    img.save('/tmp/test.png')
    
    print('[4] Testing process_image with predict()...')
    signal.alarm(60)  # 60s timeout for processing
    result = pipeline.process_image('/tmp/test.png', 'bottle')
    signal.alarm(0)  # Cancel alarm
    
    print(f'[SUCCESS] Result: {result}')
    if result['success']:
        print('✓ predict() works without CUDA errors!')
    else:
        print(f'✗ predict() failed: {result.get("error", "unknown error")}')
    
except TimeoutError as e:
    print(f'[TIMEOUT] {e}')
    sys.exit(1)
except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)
