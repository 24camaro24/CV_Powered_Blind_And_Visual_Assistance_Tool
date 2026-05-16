#!/usr/bin/env python3
"""Test predict() directly to see the actual CUDA error."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.insert(0, 'backend')

from groundingdino.util.inference import predict, load_model, load_image
from pathlib import Path
import torch

print('[TEST] Loading GroundingDINO model...')
base_dir = Path('.')
config_path = str(base_dir / 'GroundingDINO_SwinT_OGC.py')
model_path = str(base_dir / 'groundingdino_swint_ogc.pth')

try:
    model = load_model(config_path, model_path)
    model.eval()
    model = model.to('cpu')
    print('[SUCCESS] Model loaded and on CPU')
except Exception as e:
    print(f'[ERROR] Model load failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now test predict with a dummy image
print('[TEST] Creating dummy image...')
from PIL import Image
img = Image.new('RGB', (640, 480), color='blue')
img.save('test_image.png')

print('[TEST] Loading image...')
try:
    image_source, image_tensor = load_image('test_image.png')
    print(f'[SUCCESS] Image loaded, tensor on device: {image_tensor.device}')
except Exception as e:
    print(f'[ERROR] Image load failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Move tensor to CPU
if hasattr(image_tensor, 'to'):
    image_tensor = image_tensor.to('cpu')
    print(f'[SUCCESS] Image tensor moved to CPU: {image_tensor.device}')

print('[TEST] Calling predict()...')
try:
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption='bottle',
        box_threshold=0.3,
        text_threshold=0.25
    )
    print(f'[SUCCESS] Predict succeeded! Found {len(boxes)} boxes')
except Exception as e:
    print(f'[ERROR] Predict failed: {type(e).__name__}')
    print(f'Message: {e}')
    import traceback
    traceback.print_exc()
