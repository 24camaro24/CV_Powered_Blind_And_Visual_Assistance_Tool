#!/usr/bin/env python3
import sys
sys.path.insert(0, 'backend')

print("[TEST] Loading pipeline module...")
from pipeline import NavigationPipeline
print("[TEST] Pipeline module loaded!")

print("\n[TEST] Creating pipeline instance...")
p = NavigationPipeline()
print("[TEST] Pipeline created successfully!")
print(f"[TEST] GroundingDINO model: {type(p.grounding_model)}")
print(f"[TEST] All models loaded and ready!")
