#!/usr/bin/env python3
"""Quick test of fixed target extraction"""

import sys
sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/Project_Folder_GC3/CV_Powered_Blind_And_Visual_Assistance_Tool/CV_Powered_Blind_And_Visual_Assistance_Tool/quick_depth_estimation_demo_code/backend')

from pipeline import NavigationPipeline

print("\n" + "="*70)
print("TESTING FIXED TARGET EXTRACTION")
print("="*70 + "\n")

# Initialize pipeline
try:
    print("Initializing pipeline...")
    pipeline = NavigationPipeline()
    print("✓ Pipeline ready\n")
except Exception as e:
    print(f"Error initializing: {e}")
    sys.exit(1)

# Test cases
test_cases = [
    ("Can you find my water bottle?", "water bottle"),
    ("Find my keys", "keys"),
    ("Where is my phone?", "phone"),
    ("Can you locate my laptop?", "laptop"),
    ("Show me the book on the table", "book table"),  # or similar
    ("I need to find my wallet", "wallet"),
    ("Where are my glasses?", "glasses"),
    ("Find my shoes", "shoes"),
    ("What is that?", "that"),  # Tricky - "that" is pronoun
    ("Can you find the remote control?", "remote control"),
]

print("Running tests:")
print("-" * 70)

passed = 0
failed = 0

for input_text, expected in test_cases:
    result = pipeline.extract_target_from_text(input_text)
    
    # Check if result matches expected
    match = (result.lower() == expected.lower() or 
             expected.lower() in result.lower() or 
             result.lower() in expected.lower())
    
    status = "✅ PASS" if match else "❌ FAIL"
    if match:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | Input: '{input_text}'")
    print(f"      | Expected: '{expected}'")
    print(f"      | Got: '{result}'")
    print()

print("-" * 70)
print(f"\nResults: {passed} PASSED, {failed} FAILED out of {len(test_cases)} tests")
if failed == 0:
    print("✅ All tests passed!")
else:
    print(f"⚠ {failed} test(s) need attention")

print("\n" + "="*70 + "\n")
