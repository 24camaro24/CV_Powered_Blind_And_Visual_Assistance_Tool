FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for TTS and audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/requirements.txt ./backend/
COPY backend/app.py ./backend/
COPY backend/pipeline.py ./backend/
COPY GroundingDINO_SwinT_OGC.py ./
COPY groundingdino_swint_ogc.pth ./

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy frontend files
COPY frontend ./frontend/

# Create uploads directory
RUN mkdir -p /app/backend/uploads

# Expose ports
EXPOSE 5000

# Set working directory
WORKDIR /app/backend

# Start Flask server
CMD ["python", "app.py"]
