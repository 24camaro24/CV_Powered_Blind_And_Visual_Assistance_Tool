# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER BROWSER                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Frontend (Vanilla JavaScript)                │   │
│  │  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │   │
│  │  │  Image       │  │  Audio   │  │  Results &   │   │   │
│  │  │  Upload      │  │  Record  │  │  Visualize   │   │   │
│  │  └──────────────┘  └──────────┘  └──────────────┘   │   │
│  │  ▼ REST API (JSON) ▼                               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ HTTP/JSON
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                    FLASK BACKEND                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Endpoints                           │   │
│  │  /api/upload         - Receive images                │   │
│  │  /api/process        - Analyze & estimate            │   │
│  │  /api/transcribe     - Convert audio to text         │   │
│  │  /api/generate-inst  - Create voice instructions     │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         NavigationPipeline (pipeline.py)             │   │
│  │                                                       │   │
│  │  1. Image Processing              (PIL, OpenCV)      │   │
│  │     └─► Load & Normalize                             │   │
│  │                                                       │   │
│  │  2. Object Detection               (GroundingDINO)    │   │
│  │     └─► Get bounding box & confidence                │   │
│  │                                                       │   │
│  │  3. Depth Estimation               (Depth V2)        │   │
│  │     └─► Depth map & distance calculation             │   │
│  │                                                       │   │
│  │  4. Navigation Calculations        (NumPy)           │   │
│  │     └─► Angle & step count                           │   │
│  │                                                       │   │
│  │  5. Visualization                  (OpenCV)          │   │
│  │     └─► Bounding box, arrow, labels                  │   │
│  │                                                       │   │
│  │  6. Instruction Generation         (Flan-T5)         │   │
│  │     └─► Text instruction                             │   │
│  │                                                       │   │
│  │  7. Speech Synthesis               (Tacotron2 TTS)    │   │
│  │     └─► WAV audio file                               │   │
│  │                                                       │   │
│  │  8. Audio Transcription            (Faster Whisper)   │   │
│  │     └─► Text from speech                             │   │
│  │                                                       │   │
│  │  9. Target Object Extraction       (Flan-T5)         │   │
│  │     └─► Target name from speech                      │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Data Flow

### Upload & Process Flow

```
1. User uploads image
   └─► Browser: Multipart form data
       └─► /api/upload endpoint
           └─► Save file, return base64 preview

2. User specifies target
   └─► Option A: Type directly
       Option B: Voice → /api/transcribe → extract target

3. User clicks "Analyze"
   └─► /api/process request (filename + target)
       └─► pipeline.process_image()
           └─► GroundingDINO detect
               └─► Depth V2 estimate
                   └─► Calculate angle + steps
                       └─► Create visualization
                           └─► Return all results (JSON + base64 image)

4. Frontend displays results
   └─► Show angle, steps, distance
       └─► Display visualization image
           └─► Generate voice instruction
               └─► Display audio player
```

### Component Responsibilities

**Frontend (JavaScript)**
- File upload handling
- Audio recording (Web Audio API)
- API communication
- Result visualization
- User interface

**Backend (Flask)**
- API routing & validation
- ML pipeline orchestration
- Model loading & caching
- Result formatting
- File management

**Pipeline (Python)**
- GroundingDINO: Object detection with text
- Depth Anything V2: Monocular depth estimation
- Faster Whisper: Speech-to-text
- Flan-T5: Text generation
- OpenCV: Image visualization
- TTS: Text-to-speech

## Data Structures

### Image Upload
```json
{
  "success": true,
  "filename": "image_123.jpg",
  "image_base64": "iVBORw0KGgoAAAANS..."
}
```

### Process Result
```json
{
  "success": true,
  "target": "person",
  "angle": -15.5,
  "steps": 8.2,
  "distance_meters": 6.15,
  "depth": 10.25,
  "bbox": [x1, y1, x2, y2],
  "visualization": "iVBORw0KGgoAAAANS...",
  "confidence": 0.95,
  "processing_time": 8.34
}
```

### Navigation Instruction
```json
{
  "success": true,
  "instruction_text": "The person is 15 degrees to your right. Walk 8 steps forward.",
  "audio_base64": "//NExAAiNAIAEX...",
  "audio_format": "wav"
}
```

## Deployment Options

### Local Development
```bash
# Terminal 1: Backend
python backend/app.py

# Terminal 2: Frontend
python serve_frontend.py
```

### Docker Compose
```bash
docker-compose up
```

### Production Deployment
- Use Gunicorn/uWSGI for Flask
- Serve frontend with Nginx
- Run on GPU for better performance
- Add authentication/authorization

## Performance Characteristics

| Component | Time | Device |
|-----------|------|--------|
| Model Loading | 2-3 min | First run only |
| Image Processing | 1-2 sec | Per image |
| GroundingDINO | 2-4 sec | Per image |
| Depth Estimation | 2-3 sec | Per image |
| Total E2E | 5-10 sec | Per request |

## Scalability

- **Horizontal**: Add multiple Flask instances + load balancer
- **Vertical**: Use GPU (NVIDIA CUDA) for 5-10x speedup
- **Async**: Use Celery for long-running tasks
- **Caching**: Cache model predictions for common objects

## Security Considerations

- ✓ CORS enabled (configure for production)
- ✓ File size limits (50MB max)
- ✓ Input validation
- Todo: Add authentication
- Todo: Add rate limiting
- Todo: Sanitize inputs
- Todo: Use HTTPS in production

