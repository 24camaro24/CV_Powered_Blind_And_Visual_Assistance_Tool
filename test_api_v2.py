#!/usr/bin/env python3
"""Test API with a proper image file"""

import requests
import json
from PIL import Image, ImageDraw
import time

time.sleep(1)

# Create a more realistic test image with some content
print('[1] Creating test image with content...')
img = Image.new('RGB', (800, 600), color=(200, 200, 200))
draw = ImageDraw.Draw(img)

# Draw a simple bottle-like shape
draw.rectangle([300, 150, 500, 450], fill=(100, 100, 200), outline=(0, 0, 0), width=2)
draw.ellipse([350, 100, 450, 150], fill=(100, 100, 200), outline=(0, 0, 0), width=2)

img_path = '/tmp/test_bottle.jpg'
img.save(img_path)
print(f'     Image saved to {img_path}')

# Try uploading to port 5001
print('[2] Uploading to http://localhost:5001...')
try:
    with open(img_path, 'rb') as f:
        files = {'image': ('test_bottle.jpg', f, 'image/jpeg')}
        headers = {}
        
        response = requests.post(
            'http://localhost:5001/api/upload',
            files=files,
            headers=headers,
            timeout=15
        )
    
    print(f'     Status: {response.status_code}')
    print(f'     Headers: {dict(response.headers)}')
    print(f'     Body length: {len(response.content)}')
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f'     ✓ Upload successful!')
            filename = data['filename']
            print(f'     Filename: {filename}')
            
            # Process
            print('[3] Processing image...')
            payload = {'filename': filename, 'target': 'bottle'}
            
            resp2 = requests.post(
                'http://localhost:5001/api/process',
                json=payload,
                timeout=60
            )
            print(f'     Status: {resp2.status_code}')
            
            if resp2.status_code == 200:
                result = resp2.json()
                print(f'     ✓ Processing complete!')
                print(json.dumps(result, indent=2)[:500])
            else:
                print(f'     ✗ Error: {resp2.status_code}')
                print(resp2.text[:300])
                
        except json.JSONDecodeError as e:
            print(f'     ✗ Response not JSON: {response.text[:500]}')
    else:
        print(f'     ✗ Upload failed: {response.status_code}')
        print(f'     Response: {response.text[:500]}')

except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}')

print('[DONE]')
