# Voice-Guided Navigation Assistant

A Flask + JavaScript web application that uses voice commands and image processing to guide users to targets. The system detects objects in images, estimates depth, and provides navigation instructions with angle and step count.

## Features

- 📸 **Image Upload**: Upload images or take photos
- 🎤 **Voice Input**: Speak to specify target objects (automatic speech-to-text)
- 🎯 **Object Detection**: Uses GroundingDINO to detect targets in images
- 📏 **Depth Estimation**: Uses Depth Anything V2 for accurate depth maps
- 🧭 **Navigation**: Calculates angle and step count to reach target
- 🗣️ **Voice Instructions**: Generates spoken navigation instructions via TTS
- 📊 **Visualization**: Shows bounding box, arrows, and navigation data

## System Architecture

```
Backend (Flask):
├── pipeline.py        - Core ML pipeline (GroundingDINO, Depth V2, Whisper, Flan-T5, TTS)
├── app.py            - Flask REST API server
└── requirements.txt  - Python dependencies

Frontend (Vanilla JavaScript):
├── index.html        - UI with drag-drop, audio recording
└── app.js           - Client-side logic and API integration
```

## Technologies Used

### Backend Models
- **GroundingDINO**: Object detection with text prompts
- **Depth Anything V2**: Depth estimation for distance calculation
- **Faster Whisper**: Speech-to-text (small model, CPU-friendly)
- **Flan-T5 Small**: Text generation for instructions and object extraction
- **TTS (Tacotron2)**: Text-to-speech for spoken instructions

### Frontend
- Vanilla JavaScript (no framework required)
- HTML5 Canvas & File APIs
- Web Audio API for recording

## Installation

### 1. Prerequisites
- Python 3.8+
- Modern web browser (Chrome, Firefox, Safari, Edge)

### 2. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# The GroundingDINO model files should be in the parent directory:
# - groundingdino_swint_ogc.pth
# - GroundingDINO_SwinT_OGC.py
```

### 3. Frontend Setup

No installation needed! The frontend is vanilla JavaScript. Just open `frontend/index.html` in a browser or serve it with a simple HTTP server.

## Running the Application

### Step 1: Start the Flask Backend

```bash
cd backend
python app.py
```

The Flask server will start on `http://localhost:5000`

**Output:**
```
Loading GroundingDINO...
Loading Depth Anything V2...
Loading Whisper...
Loading Flan-T5...
Loading TTS...
All models loaded successfully!
 * Running on http://0.0.0.0:5000
```

### Step 2: Open the Frontend

Option A: Simple HTTP Server
```bash
cd frontend
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

Option B: Direct File Open
- Simply open `frontend/index.html` in your browser

## API Endpoints

### Health Check
- **GET** `/health` - Check if backend is running

### Image Upload
- **POST** `/api/upload`
- Body: `multipart/form-data` with `image` file
- Returns: `{ success, filename, image_base64 }`

### Process Image
- **POST** `/api/process`
- Body: `{ filename, target }`
- Returns: `{ success, angle, steps, distance_meters, visualization, ... }`

### Transcribe Audio
- **POST** `/api/transcribe`
- Body: `multipart/form-data` with `audio` file
- Returns: `{ success, transcribed_text, target }`

### Generate Instruction
- **POST** `/api/generate-instruction`
- Body: `{ target, steps, angle }`
- Returns: `{ success, instruction_text, audio_base64 }`

## Usage Workflow

1. **Upload Image**: Click the upload box or drag an image
2. **Specify Target**: 
   - Type the object name directly, OR
   - Click "Start Recording" and speak the target
3. **Analyze**: Click "Analyze & Estimate Navigation"
4. **View Results**: 
   - See angle, distance, and step count
   - Visual preview with bounding box and arrow
   - Click play to hear spoken instructions

## Example Targets

- "person"
- "chair"
- "phone"
- "cup"
- "dog"
- Any object that can be described in one or two words

## Performance Notes

- **First Run**: Model loading takes 2-3 minutes (cached afterward)
- **Image Processing**: 5-15 seconds per image (depends on image size)
- **CPU-Friendly**: All models optimized for CPU inference

## Troubleshooting

### Backend won't start
```
Error: ModuleNotFoundError: No module named 'groundingdino'
```
- Run `pip install -r requirements.txt` in the backend folder

### CORS errors in browser console
- Make sure Flask backend is running on `http://localhost:5000`
- The frontend automatically configures CORS

### No image is detected
- Try a different object name
- Ensure the object is clearly visible in the image
- Adjust text_threshold in `pipeline.py` (lower = more detections)

### Audio recording not working
- Check browser microphone permissions
- Use HTTPS (required for microphone in most browsers except localhost)

### Models are slow to load first time
- Models are downloaded from HuggingFace on first run
- Subsequent runs use cached models
- This is expected behavior

## File Structure

```
quick_depth_estimation_demo_code/
├── backend/
│   ├── app.py                    # Flask server
│   ├── pipeline.py              # ML pipeline
│   ├── requirements.txt          # Dependencies
│   └── uploads/                 # Uploaded images (created at runtime)
├── frontend/
│   ├── index.html               # Main UI
│   └── app.js                   # Client logic
├── groundingdino_swint_ogc.pth  # Model weights
├── GroundingDINO_SwinT_OGC.py   # Model config
└── README.md                     # This file
```

## Customization

### Adjust Detection Sensitivity
Edit `backend/pipeline.py`, line 83-84:
```python
boxes, logits, phrases = predict(
    box_threshold=0.3,      # Lower = more detections
    text_threshold=0.25,    # Lower = more matches
)
```

### Change Step Length Estimation
Edit `backend/pipeline.py`, line 108:
```python
steps = meters / 0.75  # Adjust 0.75 for your average step length
```

### Customize Instructions
Edit the prompt in `backend/pipeline.py`, `generate_instruction()` method

### Change UI Colors
Edit CSS in `frontend/index.html`, specifically the color variables in the `<style>` tag

## License

This project uses models from HuggingFace. Please refer to their respective licenses:
- GroundingDINO
- Depth Anything V2
- Faster Whisper
- Flan-T5
- TacotronTTS

## Support

For issues or questions, check:
1. That all Python dependencies are installed
2. That the backend is running when using the frontend
3. Browser console (F12) for error messages
4. Model download logs during first run

