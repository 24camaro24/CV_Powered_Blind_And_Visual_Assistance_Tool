#!/usr/bin/env python3
"""Test importing and initializing NavigationPipeline"""

import sys
sys.path.insert(0, 'backend')

print('[1] Importing pipeline module...')
try:
    from pipeline import NavigationPipeline
    print('[SUCCESS] Pipeline module imported')
except Exception as e:
    print(f'[ERROR] Pipeline import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('[2] Creating pipeline instance...')
try:
    pipeline = NavigationPipeline()
    print('[SUCCESS] Pipeline initialized')
except Exception as e:
    print(f'[ERROR] Pipeline init failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('[3] All checks passed!')
