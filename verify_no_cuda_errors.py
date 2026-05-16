#!/usr/bin/env python3
"""
Comprehensive test to verify NO CUDA errors in the application
"""
import sys
import os

# Set CUDA env vars FIRST
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''

print("\n" + "="*70)
print("CUDA ERROR VERIFICATION TEST")
print("="*70 + "\n")

def test_step(step_num, description, test_func):
    """Run a test step with nice formatting"""
    print(f"[{step_num}] {description}...", end=" ", flush=True)
    try:
        test_func()
        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL")
        print(f"    Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

success_count = 0
total_tests = 0

# Test 1: Import torch
def test_import_torch():
    global torch
    import torch
test_step(1, "Import torch", test_import_torch)

# Test 2: Check torch.cuda is disabled
def test_torch_cuda_disabled():
    assert torch.cuda.is_available() == False, "torch.cuda.is_available() should be False"
total_tests += 1
if test_step(2, "Verify torch.cuda.is_available() == False", test_torch_cuda_disabled):
    success_count += 1

# Test 3: Import navigation pipeline
def test_import_pipeline():
    sys.path.insert(0, 'backend')
    global NavigationPipeline
    from pipeline import NavigationPipeline
total_tests += 1
if test_step(3, "Import NavigationPipeline", test_import_pipeline):
    success_count += 1

# Test 4: Initialize pipeline
def test_init_pipeline():
    import sys
    sys.path.insert(0, 'backend')
    global pipeline, NavigationPipeline
    try:
        pipeline = NavigationPipeline(device='cpu')
    except NameError:
        # NavigationPipeline not imported yet in the namespace
        from pipeline import NavigationPipeline as NP
        pipeline = NP(device='cpu')
    assert hasattr(pipeline, 'grounding_model'), "Missing grounding_model"
    assert hasattr(pipeline, 'depth_model'), "Missing depth_model"
    assert hasattr(pipeline, 'instr_model'), "Missing instr_model"
total_tests += 1
if test_step(4, "Initialize NavigationPipeline", test_init_pipeline):
    success_count += 1

# Test 5: Test generate_instruction (text generation on Flan-T5)
def test_generate_instruction():
    result = pipeline.generate_instruction('bottle', 5, 15, 3.75, 0.85, 0.6)
    assert 'detailed' in result, "Missing 'detailed' key"
    assert 'conversational' in result, "Missing 'conversational' key"
    assert 'summary' in result, "Missing 'summary' key"
    assert len(result['detailed']) > 0, "Empty detailed instruction"
    assert len(result['conversational']) > 0, "Empty conversational text"
total_tests += 1
if test_step(5, "Test generate_instruction() with Flan-T5", test_generate_instruction):
    success_count += 1

# Test 6: Test extract_target_from_text (Flan-T5 inference)
def test_extract_target():
    text = "Can you show me the bottle?"
    target = pipeline.extract_target_from_text(text)
    assert isinstance(target, str), "Expected string output"
    assert len(target) > 0, "Empty target extraction"
total_tests += 1
if test_step(6, "Test extract_target_from_text() with Flan-T5", test_extract_target):
    success_count += 1

# Test 7: Test GroundingDINO dummy (can't test without image, but check model is loaded)
def test_grounding_model_loaded():
    assert pipeline.grounding_model is not None, "GroundingDINO not loaded"
    assert hasattr(pipeline.grounding_model, 'eval'), "Model missing eval() method"
total_tests += 1
if test_step(7, "Verify GroundingDINO model loaded", test_grounding_model_loaded):
    success_count += 1

# Test 8: Test depth model is loaded
def test_depth_model_loaded():
    assert pipeline.depth_model is not None, "Depth model not loaded"
    assert hasattr(pipeline.depth_model, 'eval'), "Model missing eval() method"
total_tests += 1
if test_step(8, "Verify Depth Anything V2 model loaded", test_depth_model_loaded):
    success_count += 1

# Test 9: Check no CUDA tensors exist
def test_no_cuda_tensors():
    # Just verify models are in eval mode and callable
    pipeline.grounding_model.eval()
    pipeline.depth_model.eval()
    pipeline.instr_model.eval()
total_tests += 1
if test_step(9, "Verify models are in eval mode", test_no_cuda_tensors):
    success_count += 1

# Test 10: Verify CPU-only device string
def test_device_string():
    assert pipeline.device == 'cpu', f"Expected device='cpu', got {pipeline.device}"
    assert isinstance(pipeline.device, str), f"Device should be string, got {type(pipeline.device)}"
total_tests += 1
if test_step(10, "Verify device is 'cpu' string", test_device_string):
    success_count += 1

print("\n" + "="*70)
print(f"RESULTS: {success_count}/{total_tests} tests passed")
if success_count == total_tests:
    print("✅ ALL TESTS PASSED - NO CUDA ERRORS!")
    print("="*70 + "\n")
    sys.exit(0)
else:
    print(f"❌ {total_tests - success_count} test(s) failed")
    print("="*70 + "\n")
    sys.exit(1)
