#!/usr/bin/env python3
"""Quick test of pipeline without server"""
import sys
import os

# Disable CUDA first
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''

sys.path.insert(0, 'backend')

try:
    from pipeline import NavigationPipeline
    
    print("=" * 60)
    print("Testing NavigationPipeline")
    print("=" * 60)
    
    print("\n[1] Initializing pipeline...")
    pipeline = NavigationPipeline()
    print("✅ Pipeline initialized successfully\n")
    
    print("[2] Testing generate_instruction()...")
    result = pipeline.generate_instruction('bottle', 5, 15, 3.75, 0.85, 0.6)
    print("✅ generate_instruction() works!")
    print(f"   - Detailed text length: {len(result['detailed'])} chars")
    print(f"   - Conversational text: {result['conversational'][:40]}...")
    print(f"   - Summary: {result['summary']}\n")
    
    print("[3] Testing extract_target_from_text()...")
    text = "Can you show me where the water bottle is?"
    target = pipeline.extract_target_from_text(text)
    print(f"✅ extract_target_from_text() works!")
    print(f"   - Input: '{text}'")
    print(f"   - Extracted target: '{target}'\n")
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED - No CUDA errors!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"Message: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
