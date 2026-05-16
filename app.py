import os
import sys

# CRITICAL: Disable CUDA BEFORE ANY imports that might use torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Now safe to import everything else
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import time

# Add backend folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pipeline import NavigationPipeline

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'backend/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize pipeline
print("Initializing Navigation Pipeline...")
pipeline = NavigationPipeline(device='cpu')
print("Pipeline ready!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and encode image to base64
        with open(filepath, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_base64': img_base64
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process():
    """Process image and estimate navigation parameters"""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data or 'target' not in data:
            return jsonify({'error': 'Missing filename or target'}), 400
        
        filename = data['filename']
        target = data['target']
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Process the image
        start_time = time.time()
        result = pipeline.process_image(filepath, target)
        elapsed = time.time() - start_time
        
        if not result['success']:
            return jsonify({'error': result['error']}), 400
        
        result['processing_time'] = elapsed
        
        # Add comprehensive instruction data to the response
        instruction_result = pipeline.generate_instruction(
            result['target'],
            result['steps'],
            result['angle'],
            result['distance_meters'],
            result.get('confidence', 0.85),
            result.get('depth', 0),
            result.get('surfaces', None)  # Pass surface context to instruction generator
        )
        
        result['navigation_guidance'] = {
            'detailed_text': instruction_result['detailed'],
            'conversational_text': instruction_result['conversational'],
            'summary': instruction_result['summary']
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Transcribe
            transcribed_text = pipeline.transcribe_audio(tmp_path)
            
            # Extract target object
            target = pipeline.extract_target_from_text(transcribed_text)
            
            return jsonify({
                'success': True,
                'transcribed_text': transcribed_text,
                'target': target
            }), 200
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-instruction', methods=['POST'])
def generate_instruction():
    """Generate comprehensive spoken instruction"""
    try:
        data = request.get_json()
        
        if not data or 'target' not in data or 'steps' not in data or 'angle' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        target = data['target']
        steps = float(data['steps'])
        angle = float(data['angle'])
        distance_meters = data.get('distance_meters', steps * 0.75)
        confidence = data.get('confidence', 0.85)
        depth = data.get('depth', 0)
        
        # Generate comprehensive instruction
        instruction_result = pipeline.generate_instruction(
            target, steps, angle, distance_meters, confidence, depth
        )
        
        # Generate audio from conversational version
        audio_path = pipeline.text_to_speech(instruction_result['conversational'])
        
        # Convert audio to base64
        with open(audio_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'detailed_text': instruction_result['detailed'],
            'conversational_text': instruction_result['conversational'],
            'summary': instruction_result['summary'],
            'audio_base64': audio_base64,
            'audio_format': 'wav'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
