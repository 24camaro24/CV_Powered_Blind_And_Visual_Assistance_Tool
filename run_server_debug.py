#!/usr/bin/env python3
"""Debug wrapper for running the server with detailed logging"""
import os
import sys
import signal

# Set up timeout handler
def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Server initialization took too long, forcefully shutting down", flush=True)
    sys.exit(1)

# 60 second timeout for total initialization
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

print("[SERVER] Starting Flask server wrapper...", flush=True)
print("[SERVER] Python version:", sys.version.split()[0], flush=True)
print("[SERVER] Working directory:", os.getcwd(), flush=True)

try:
    print("\n[SERVER] Phase 1: Importing Flask and other dependencies...", flush=True)
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    import base64
    import time
    from pathlib import Path
    print("[SERVER] ✓ Flask and dependencies imported", flush=True)
    
    print("\n[SERVER] Phase 2: Adding backend to sys.path...", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    print("[SERVER] ✓ Backend path added", flush=True)
    
    print("\n[SERVER] Phase 3: Importing pipeline module...", flush=True)
    from pipeline import NavigationPipeline
    print("[SERVER] ✓ Pipeline module imported", flush=True)
    
    print("\n[SERVER] Phase 4: Creating Flask app...", flush=True)
    app = Flask(__name__, static_folder='frontend', static_url_path='')
    CORS(app)
    print("[SERVER] ✓ Flask app created", flush=True)
    
    print("\n[SERVER] Phase 5: Configuring upload settings...", flush=True)
    UPLOAD_FOLDER = 'backend/uploads'
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    print("[SERVER] ✓ Upload configured", flush=True)
    
    print("\n[SERVER] Phase 6: Initializing NavigationPipeline...", flush=True)
    print("[SERVER]   This may take 30-60 seconds for model loading...", flush=True)
    pipeline = NavigationPipeline(device='cpu')
    print("[SERVER] ✓ Pipeline initialized", flush=True)
    
    print("\n[SERVER] Phase 7: Setting up routes...", flush=True)
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    def index():
        return send_from_directory('frontend', 'index.html')

    @app.route('/<path:filename>')
    def serve_static(filename):
        try:
            return send_from_directory('frontend', filename)
        except:
            return send_from_directory('frontend', 'index.html')

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

    @app.route('/api/upload', methods=['POST'])
    def upload_image():
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            with open(filepath, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()
            return jsonify({'success': True, 'filename': filename, 'image_base64': img_base64}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/process', methods=['POST'])
    def process():
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image provided'}), 400
            result = pipeline.process(data['image'], data.get('target', 'bottle'))
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    print("[SERVER] ✓ Routes configured", flush=True)
    
    # Cancel the alarm since we're done initializing
    signal.alarm(0)
    
    print("\n" + "="*60, flush=True)
    print("Voice-Guided Navigation Assistant", flush=True)
    print("="*60, flush=True)
    print("Backend + Frontend Server starting...", flush=True)
    print("Open your browser to: http://localhost:5001", flush=True)
    print("Press Ctrl+C to stop the server", flush=True)
    print("="*60 + "\n", flush=True)
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
    
except Exception as e:
    print(f"\n[ERROR] Initialization failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)