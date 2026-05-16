#!/usr/bin/env python3
"""Test pipeline import with timeout"""

import signal
import sys
import os

os.chdir('/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/quick_depth_estimation_demo_code')
sys.path.insert(0, 'backend')

def timeout_handler(signum, frame):
    raise TimeoutError("Pipeline initialization took too long")

# Set 30 second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

try:
    print('[1] Importing pipeline...')
    from pipeline import NavigationPipeline
    
    print('[2] Creating pipeline...')
    p = NavigationPipeline()
    
    print('[3] SUCCESS - Pipeline created!')
    signal.alarm(0)  # Cancel alarm
    
except TimeoutError:
    print('[TIMEOUT] Pipeline initialization took too long (>30s)')
    sys.exit(1)
except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)
