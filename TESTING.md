# Testing Guide

## Unit Testing the Pipeline

Create `backend/test_pipeline.py`:

```python
import unittest
from pathlib import Path
from pipeline import NavigationPipeline

class TestNavigationPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load models once for all tests"""
        print("\n🔄 Loading models...")
        cls.pipeline = NavigationPipeline(device='cpu')
        print("✓ Models loaded")
    
    def test_model_loading(self):
        """Test that all models load successfully"""
        self.assertIsNotNone(self.pipeline.grounding_model)
        self.assertIsNotNone(self.pipeline.depth_model)
        self.assertIsNotNone(self.pipeline.whisper_model)
        self.assertIsNotNone(self.pipeline.instr_model)
        self.assertIsNotNone(self.pipeline.tts)
        print("✓ All models loaded")
    
    def test_text_extraction(self):
        """Test object extraction from text"""
        text = "I want to go to the chair in the room"
        target = self.pipeline.extract_target_from_text(text)
        self.assertIsNotNone(target)
        self.assertTrue(len(target) > 0)
        print(f"✓ Extracted target: {target}")
    
    def test_instruction_generation(self):
        """Test instruction text generation"""
        text = self.pipeline.generate_instruction("person", 10, 45)
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)
        self.assertIn("person", text.lower())
        print(f"✓ Generated instruction: {text}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

Run tests:
```bash
cd backend
python test_pipeline.py
```

## Integration Testing

### Test 1: Image Processing with Sample Image

```python
# backend/test_sample.py
from pipeline import NavigationPipeline
from PIL import Image
import numpy as np

# Create a simple test image
img = Image.new('RGB', (640, 480), color='white')
img.save('test_image.jpg')

# Load pipeline
pipeline = NavigationPipeline(device='cpu')

# Process
result = pipeline.process_image('test_image.jpg', 'person')
print("Result:", result)
```

### Test 2: API Endpoint Testing

```bash
# Test health check
curl http://localhost:5000/health

# Test upload
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/upload

# Test process
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"filename":"test_image.jpg","target":"person"}'
```

### Test 3: Frontend Testing

Manual testing checklist:
- [ ] Image upload works
- [ ] Image preview displays
- [ ] Text input works
- [ ] Audio recording starts/stops
- [ ] Transcribed text appears
- [ ] Process button triggers analysis
- [ ] Results display correctly
- [ ] Audio playback works
- [ ] Error messages display properly

## Performance Testing

### Measure Processing Time

```python
import time
from pipeline import NavigationPipeline

pipeline = NavigationPipeline()

# Test with various image sizes
images = ['small.jpg', 'medium.jpg', 'large.jpg']

for img_path in images:
    start = time.time()
    result = pipeline.process_image(img_path, 'person')
    elapsed = time.time() - start
    print(f"{img_path}: {elapsed:.2f}s")
```

### Load Testing

```bash
# Install apache bench
brew install httpd  # macOS

# Run load test
ab -n 100 -c 10 http://localhost:5000/health
```

## Browser Console Testing

Open browser DevTools (F12) → Console and run:

```javascript
// Test API connectivity
fetch('http://localhost:5000/health')
  .then(r => r.json())
  .then(d => console.log('Backend OK:', d))
  .catch(e => console.error('Backend error:', e));

// Test video recording capability
navigator.mediaDevices.enumerateDevices()
  .then(devices => console.log('Devices:', devices));
```

## Common Issues & Solutions

### Issue: "Target not detected"
**Test:**
```bash
# Try with a specific target
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"filename":"image.jpg","target":"face"}'
```

**Solution:** Adjust detection thresholds in `pipeline.py`:
```python
box_threshold=0.2,  # Lower = more sensitive
text_threshold=0.2  # Lower = more matches
```

### Issue: "Connection refused"
**Test:**
```bash
# Check if backend is running
netstat -an | grep 5000
# Or
lsof -i :5000
```

**Solution:** Start backend with:
```bash
python backend/app.py
```

### Issue: Slow processing
**Test:**
```python
import time
start = time.time()
# Run process_image
print(f"Time: {time.time() - start}s")
```

**Solutions:**
1. Use GPU: Install CUDA & torch-gpu
2. Use smaller images
3. Reduce model quality (faster models)

### Issue: Out of memory
**Solution:**
```python
# Reduce batch size or model size in pipeline.py
# Use lighter models like depth_anything_v2_small
```

## Continuous Integration

Setup GitHub Actions (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r backend/requirements.txt
      - run: cd backend && python test_pipeline.py
```

## Screenshot Testing

For visual regression testing, save expected outputs:

```python
from PIL import Image
import base64

# Save base64 image for comparison
with open('backend/uploads/image.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()
    
# Compare with expected
expected = "iVBORw0KGgoAAAANS..."
assert img_base64 == expected
```

## Monitoring

Add simple logging:

```python
# backend/app.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/process', methods=['POST'])
def process():
    logger.info("Processing request")
    # ...
    logger.info(f"Result: {result}")
```

View logs:
```bash
python backend/app.py 2>&1 | tee app.log
```

