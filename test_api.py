#!/usr/bin/env python3
"""Test the actual API by uploading and processing an image"""

import requests
import json
from PIL import Image
import time

# Give server time if needed
time.sleep(1)

# Create a simple test image
print('[1] Creating test image...')
img = Image.new('RGB', (640, 480), color=(100, 150, 200))
img.save('/tmp/test_image.png')
print('     Test image saved')

# Upload image
print('[2] Uploading image to /api/upload...')
try:
    with open('/tmp/test_image.png', 'rb') as f:
        files = {'image': f}
        # Try both ports
        for port in [5001, 5000]:
            try:
                response = requests.post(f'http://localhost:{port}/api/upload', files=files, timeout=10)
                print(f'     Using port {port}')
                break
            except:
                continue
    
    print(f'     Response status: {response.status_code}')
    
    if response.status_code == 200:
        upload_data = response.json()
        filename = upload_data['filename']
        print(f'     ✓ Image uploaded: {filename}')
        
        # Process image
        print('[3] Processing image with /api/process...')
        process_payload = {
            'filename': filename,
            'target': 'object'
        }
        
        try:
            response = requests.post(
                f'http://localhost:{port}/api/process',
                json=process_payload,
                timeout=60
            )
            print(f'     Response status: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                print('[✓] Processing completed!')
                print(f'     Result: {json.dumps(result, indent=2)[:500]}...')
            else:
                print(f'[✗] Processing failed: {response.status_code}')
                print(f'     Error: {response.text[:200]}')
                
        except requests.exceptions.Timeout:
            print('[⏱] API call timed out after 60 seconds')
        except Exception as e:
            print(f'[ERROR] API call failed: {e}')
    else:
        print(f'     ✗ Upload failed: {response.status_code}')
        print(f'     Response: {response.text[:200]}')
        
except ConnectionRefusedError:
    print('[ERROR] Could not connect to server')
except Exception as e:
    print(f'[ERROR] Test failed: {e}')

print('[DONE]')

