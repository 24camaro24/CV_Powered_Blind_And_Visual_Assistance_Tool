import os
import sys

# CRITICAL: Disable CUDA BEFORE ANY imports that might use torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Disable HuggingFace model downloads if cache not available
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

# Now safe to import everything else
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import time
import numpy as np
from pathlib import Path

# Add backend folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Ensure we're in the correct working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pipeline import NavigationPipeline

app = Flask(__name__, static_folder='frontend', static_url_path='')
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
print("="*60)
print("Initializing Navigation Pipeline...")
try:
    pipeline = NavigationPipeline(device='cpu')
    print("✅ Pipeline initialized successfully!")
    print("   All models loaded on CPU")
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"[WARNING] CUDA error during initialization: {e}")
        print("[INFO] Attempting to continue with CPU-only mode...")
        pipeline = NavigationPipeline(device='cpu')
    else:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        raise
except Exception as e:
    print(f"[ERROR] Unexpected error during initialization: {e}")
    raise
print("="*60)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== STATIC FILES ====================
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend folder"""
    try:
        return send_from_directory('frontend', filename)
    except:
        # If static file not found, serve index.html (for single-page app)
        return send_from_directory('frontend', 'index.html')

# ==================== API ENDPOINTS ====================
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
        
        # Process the image with explicit error handling
        try:
            start_time = time.time()
            result = pipeline.process_image(filepath, target)
            elapsed = time.time() - start_time
            
            if not result['success']:
                print(f"[API] process_image failed: {result['error']}")
                return jsonify({'error': result['error']}), 400
            
            result['processing_time'] = elapsed
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[API ERROR] CUDA error in process_image: {e}")
                return jsonify({'error': f'Model processing error (CUDA): {str(e)}'}, ), 500
            else:
                raise
        
        # Generate instruction with explicit error handling
        try:
            instruction_result = pipeline.generate_instruction(
                result['target'],
                result['steps'],
                result['angle'],
                result['distance_meters'],
                result.get('confidence', 0.85),
                result.get('depth', 0),
                result.get('surfaces', None)  # Pass spatial relationship data
            )
            
            result['navigation_guidance'] = {
                'detailed_text': instruction_result['detailed'],
                'conversational_text': instruction_result['conversational'],
                'summary': instruction_result['summary']
            }
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[API ERROR] CUDA error in generate_instruction: {e}")
                # Still return the result but without guidance
                surfaces = result.get('surfaces', [])
                surface_info = f" on {surfaces[0]['surface']}" if surfaces else ""
                
                result['navigation_guidance'] = {
                    'detailed_text': f'Navigation to {result["target"]}{surface_info}',
                    'conversational_text': f'Target found at {result["distance_meters"]:.1f} meters{surface_info}',
                    'summary': {
                        'target': result['target'],
                        'distance_m': round(result['distance_meters'], 2),
                        'distance_ft': round(result['distance_meters'] * 3.28, 2),
                        'steps': int(result['steps']),
                        'direction': 'ahead',
                        'angle_degrees': round(result['angle'], 1),
                        'confidence_percent': round(result.get('confidence', 0.85) * 100, 1),
                        'depth_m': round(result.get('depth', 0), 3),
                        'on_surface': surfaces[0]['surface'] if surfaces else None
                    }
                }
            else:
                raise
        
        return jsonify(result), 200
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print(f"[API FATAL] Unhandled CUDA error: {e}")
            return jsonify({'error': f'CUDA processing error: Please ensure models are on CPU'}), 500
        else:
            raise
    except Exception as e:
        print(f"[API ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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
        surfaces = data.get('surfaces', None)  # Extract surfaces from request
        
        # Generate comprehensive instruction
        instruction_result = pipeline.generate_instruction(
            target, steps, angle, distance_meters, confidence, depth, surfaces
        )
        
        # Try to generate audio, but don't block if TTS is slow
        audio_base64 = None
        try:
            audio_path = pipeline.text_to_speech(instruction_result['conversational'])
            # Convert audio to base64
            with open(audio_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode()
        except Exception as tts_error:
            print(f"[TTS] Warning - audio generation failed: {tts_error}")
            # Continue without audio
        
        return jsonify({
            'success': True,
            'detailed_text': instruction_result['detailed'],
            'conversational_text': instruction_result['conversational'],
            'summary': instruction_result['summary'],
            'audio_base64': audio_base64,
            'audio_format': 'wav' if audio_base64 else None
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== FEW-SHOT LEARNING ENDPOINTS ====================
@app.route('/api/few-shot/add-reference', methods=['POST'])
def add_few_shot_reference():
    """Add a reference image for few-shot object learning"""
    try:
        if 'image' not in request.files or 'object_name' not in request.form:
            return jsonify({'error': 'Missing image or object_name'}), 400
        
        file = request.files['image']
        object_name = request.form['object_name'].strip()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        if not object_name:
            return jsonify({'error': 'Object name cannot be empty'}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{object_name}_{filename}")
        file.save(filepath)
        
        try:
            # Load image
            from PIL import Image
            image = Image.open(filepath)
            image_array = np.array(image).astype(np.uint8)
            
            # Add to few-shot matcher
            pipeline.few_shot_matcher.add_reference(object_name, image_array)
            
            return jsonify({
                'success': True,
                'object_name': object_name,
                'message': f'Reference added for {object_name}'
            }), 200
        finally:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/few-shot/database-info', methods=['GET'])
def get_few_shot_database_info():
    """Get information about stored few-shot references"""
    try:
        if not pipeline.few_shot_matcher:
            return jsonify({
                'success': True,
                'database': {},
                'total_objects': 0,
                'message': 'FewShotMatcher not initialized'
            }), 200
        
        db_info = pipeline.few_shot_matcher.get_database_info()
        
        return jsonify({
            'success': True,
            'database': db_info,
            'total_objects': len(db_info),
            'objects': list(db_info.keys())
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/few-shot/clear-references', methods=['POST'])
def clear_few_shot_references():
    """Clear stored few-shot references"""
    try:
        data = request.get_json() or {}
        object_name = data.get('object_name', None)
        
        if not pipeline.few_shot_matcher:
            return jsonify({'error': 'FewShotMatcher not initialized'}), 400
        
        pipeline.few_shot_matcher.clear_references(object_name)
        
        if object_name:
            message = f'Cleared references for {object_name}'
        else:
            message = 'Cleared all references'
        
        return jsonify({
            'success': True,
            'message': message
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(404)
def not_found(error):
    # Try to serve static file, otherwise return 404
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Global error handler for CUDA/RuntimeErrors
@app.errorhandler(RuntimeError)
def handle_runtime_error(error):
    error_str = str(error)
    print(f"[GLOBAL] RuntimeError caught: {error_str}")
    if "CUDA" in error_str or "cuda" in error_str:
        print("[GLOBAL] CUDA error detected - returning graceful error response")
        return jsonify({'error': 'Model processing error. Ensure models are configured for CPU-only mode.'}), 500
    return jsonify({'error': error_str}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Voice-Guided Navigation Assistant")
    print("="*60)
    print("\nBackend + Frontend Server starting...")
    print("Open your browser to: http://localhost:5001")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    # Disable debug mode to avoid auto-reloader issues and hanging requests
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5001, threaded=True)
