# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

The first time you run the application, install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

This will install all required packages. The GroundingDINO model files should already be in the parent directory.

### Step 2: Start the Backend Server

Terminal 1:
```bash
python backend/app.py
```

Or use the helper script:
```bash
bash start_server.sh
```

You should see:
```
Loading GroundingDINO...
Loading Depth Anything V2...
Loading Whisper...
Loading Flan-T5...
Loading TTS...
All models loaded successfully!
 * Running on http://0.0.0.0:5000
```

### Step 3: Open the Frontend

Terminal 2:
```bash
python serve_frontend.py
```

Then open your browser to `http://localhost:8000`

**Note:** Make sure the Flask backend servers is running on port 5000 before using the frontend.

---

## 📱 Using the Application

1. **Upload an Image**
   - Click the upload box
   - Select or drag-drop an image file
   - Preview appears immediately

2. **Specify Target**
   - **Option A:** Type the object name (e.g., "person", "coffee cup")
   - **Option B:** Click "🎙️ Start Recording" → Speak → The system will transcribe and extract the target

3. **Click "🚀 Analyze & Estimate Navigation"**
   - Processing takes 5-15 seconds
   - The system will:
     - Detect the target object
     - Estimate depth/distance
     - Calculate navigation angle
     - Generate a visualization
     - Create voice instructions

4. **View Results**
   - Angle and direction (left/right)
   - Distance in meters
   - Estimated number of steps
   - Visualization with bounding box and arrow
   - Click the audio player to hear instructions

---

## 🔧 Troubleshooting

### "Connection refused" error
- Make sure Flask backend is running on port 5000
- Check: `curl http://localhost:5000/health`

### "Target not detected"
- Try a different object name (be specific)
- Ensure the object is clearly visible in the image
- Try one or two word descriptions

### Audio recording not working
- Check microphone permissions in your browser
- Some browsers require HTTPS (localhost is an exception)
- Try a different browser

### Slow processing on first run
- The ML models are being downloaded from HuggingFace
- This is normal and only happens once
- Subsequent runs will be much faster (cached models)

### Port 5000 or 8000 already in use
Change the port in:
- `backend/app.py` (line ~180): `app.run(port=5001)`
- `serve_frontend.py` (line ~16): `PORT = 8001`

---

## ⚡ Performance Tips

1. **Use smaller images** (under 1080p) for faster processing
2. **First run takes longer** - models are cached afterward
3. **CPU-friendly** - no GPU needed, but GPU will speed things up
4. **Modern browser** recommended for best performance

---

## 📚 What Each Component Does

### Pipeline (Backend)
1. **Image Processing**: Loads and prepares image
2. **GroundingDINO**: Detects the target object
3. **Depth Anything V2**: Creates depth map
4. **Calculation**: Determines angle and distance
5. **Visualization**: Draws bounding box and arrow
6. **Flan-T5**: Generates instruction text
7. **TTS**: Creates voice instruction

### Frontend
1. **Image Upload**: Handles file uploads
2. **Audio Recording**: Records and sends audio via microphone
3. **API Client**: Communicates with Flask backend
4. **Visualization Display**: Shows results with images
5. **Audio Player**: Plays voice instructions

---

## 🎯 Example Use Cases

- **Navigation**: "Show me how to reach the door"
- **Object Finding**: "Where is the emergency exit?"
- **Accessibility**: "Guide me to the person waving"
- **Delivery**: "Direct me to the package on the table"

---

## 📖 Full Documentation

See `README.md` for:
- Complete API documentation
- Architecture details
- Customization options
- Advanced configuration
