                    #!/usr/bin/env python3
"""
Quick test script to isolate the torch.device issue
"""
import sys
import traceback

print("="*60)
print("DEVICE ISSUE TEST SCRIPT")
print("="*60)

# Test 1: Check basic device string handling
print("\n[TEST 1] Basic device string handling...")
import torch
device = 'cpu'
print(f"  device = '{device}' (type: {type(device).__name__})")
print(f"  isinstance(device, str) = {isinstance(device, str)}")

# Test 2: Check if torch.device is being created
print("\n[TEST 2] torch.device creation...")
try:
    torch_device = torch.device(device)
    print(f"  torch_device = {torch_device} (type: {type(torch_device).__name__})")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Try a simple .to() call
print("\n[TEST 3] Simple tensor .to() call...")
try:
    tensor = torch.randn(2, 2)
    tensor_cpu = tensor.to('cpu')
    print(f"  ✓ tensor.to('cpu') succeeded")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Test 4: Try .to() with torch.device (this should fail)
print("\n[TEST 4] tensor.to() with torch.device (should fail)...")
try:
    tensor = torch.randn(2, 2)
    torch_device = torch.device('cpu')
    tensor_device = tensor.to(torch_device)
    print(f"  ✓ tensor.to(torch.device) succeeded (unexpected!)")
except TypeError as e:
    print(f"  ✗ Expected TypeError: {e}")
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")

# Test 5: Test NavigationPipeline initialization
print("\n[TEST 5] NavigationPipeline initialization...")
try:
    sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Intelligent_Sensing_Systems/quick_depth_estimation_demo_code/backend')
    from pipeline import NavigationPipeline
    print("  ✓ Pipeline imported successfully")
    
    print("  Creating pipeline with device='cpu'...")
    pipeline = NavigationPipeline(device='cpu')
    print("  ✓ Pipeline created successfully!")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    print("\n[FULL TRACEBACK]")
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
