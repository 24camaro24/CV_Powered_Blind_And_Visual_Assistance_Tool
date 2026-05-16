// NaviGuide - Beautiful Voice-Guided Navigation App
const API_BASE_URL = '';

class NavigationApp {
    constructor() {
        this.mode = null;  // 'blind' or 'visually-impaired'
        this.uploadedFile = null;
        this.uploadedFilename = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.videoStream = null;
        this.blindModeUsingCamera = true;  // Track which image source is used in blind mode
        this.init();
    }
    
    init() {
        this.renderModeSelection();
        this.checkBackendHealth();
    }
    
    renderModeSelection() {
        const root = document.getElementById('root');
        root.innerHTML = `
            <div class="container">
                <h1>🎯 Voice-Guided Navigation Assistant</h1>
                
                <div class="mode-selection">
                    <p style="font-size: 1.1em; color: #666; margin-bottom: 20px;">Choose your mode:</p>
                    
                    <div class="mode-card" id="blindModeCard">
                        <div class="mode-icon">🦻</div>
                        <h2>Blind Mode</h2>
                        <p>Audio-only navigation using device camera</p>
                        <p style="font-size: 0.9em; color: #999;">Speak to specify what you want to find. Audio guidance provided automatically.</p>
                    </div>
                    
                    <div class="mode-card" id="visuallyImpairedModeCard">
                        <div class="mode-icon">👁️</div>
                        <h2>Visually Impaired Mode</h2>
                        <p>Image-based navigation with audio guidance</p>
                        <p style="font-size: 0.9em; color: #999;">Upload an image and specify target. Get visual results with audio guidance.</p>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('blindModeCard').addEventListener('click', () => this.selectMode('blind'));
        document.getElementById('visuallyImpairedModeCard').addEventListener('click', () => this.selectMode('visually-impaired'));
    }
    
    selectMode(mode) {
        this.mode = mode;
        if (mode === 'blind') {
            this.renderBlindMode();
        } else {
            this.renderVisuallyImpairedMode();
        }
    }
    
    renderBlindMode() {
        const root = document.getElementById('root');
        root.innerHTML = `
            <div class="container">
                <h1>🦻 Blind Mode - Voice Navigation</h1>
                <p style="text-align: center; color: #666; margin-bottom: 30px;">Specify the object you want to find (upload image or use camera)</p>
                
                <div class="grid">
                    <!-- Image/Camera Section -->
                    <div class="section">
                        <h2>📸 Image Source</h2>
                        
                        <div style="margin-bottom: 15px;">
                            <button class="btn-primary" id="cameraBtn" style="width: 100%; margin-bottom: 10px;">
                                📷 Use Camera
                            </button>
                        </div>
                        
                        <video id="cameraFeed" class="camera-preview active" autoplay playsinline style="display: none;"></video>
                        
                        <p style="text-align: center; color: #999; margin-bottom: 15px;">OR</p>
                        
                        <div class="upload-box" id="uploadBox">
                            <p class="upload-text">Click or drag image here</p>
                            <p class="upload-hint">JPG, PNG, GIF (max 50MB)</p>
                            <input type="file" id="imageInput" accept="image/*">
                            <div id="previewContainer"></div>
                        </div>
                    </div>
                    
                    <!-- Voice Request Section -->
                    <div class="section">
                        <h2>🎤 Record Your Request</h2>
                        <p style="color: #666; font-size: 0.95em; margin-bottom: 15px;">Say what object you want to find</p>
                        
                        <button class="btn-primary" id="recordBtn" style="width: 100%; margin-bottom: 15px;">
                            <span id="recordBtnText">🎙️ Start Recording</span>
                        </button>
                        
                        <div id="audioTranscript" style="margin-top: 15px;"></div>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <button class="btn-primary" id="processBtn" style="max-width: 300px; padding: 15px 30px; font-size: 1.1em; display: none;">
                        🚀 Get Navigation Guidance
                    </button>
                </div>
                
                <button class="btn-secondary" id="backBtn" style="width: 100%; max-width: 300px; display: block; margin: 0 auto; padding: 12px 30px;">← Back to Mode Selection</button>
                
                <div id="resultsSection"></div>
            </div>
        `;
        
        this.setupBlindModeListeners();
    }
    
    renderVisuallyImpairedMode() {
        const root = document.getElementById('root');
        root.innerHTML = `
            <div class="container">
                <h1>👁️ Visually Impaired Mode - Image Navigation</h1>
                
                <div class="grid">
                    <!-- Image/Camera Section -->
                    <div class="section">
                        <h2>📸 Image Source</h2>
                        
                        <div style="margin-bottom: 15px;">
                            <button class="btn-primary" id="cameraBtn" style="width: 100%; margin-bottom: 10px;">
                                📷 Use Camera
                            </button>
                        </div>
                        
                        <video id="cameraFeed" class="camera-preview" autoplay playsinline style="display: none;"></video>
                        
                        <p style="text-align: center; color: #999; margin-bottom: 15px;">OR</p>
                        
                        <div class="upload-box" id="uploadBox">
                            <p class="upload-text">Click or drag image here</p>
                            <p class="upload-hint">JPG, PNG, GIF, BMP (max 50MB)</p>
                            <input type="file" id="imageInput" accept="image/*">
                            <div id="previewContainer"></div>
                        </div>
                    </div>
                    
                    <!-- Target Input Section -->
                    <div class="section">
                        <h2>🎤 Specify Target</h2>
                        
                        <input type="text" id="targetInput" class="target-input" placeholder="Type target object here...">
                        
                        <p style="color: #999; margin-bottom: 10px;">OR</p>
                        
                        <button class="btn-primary" id="recordBtn" style="margin-bottom: 10px; width: 100%;">
                            <span id="recordBtnText">🎙️ Start Recording</span>
                        </button>
                        
                        <div id="audioTranscript" style="margin-top: 10px;"></div>
                    </div>
                </div>
                
                <!-- Process Button -->
                <div style="text-align: center; margin: 30px 0;">
                    <button class="btn-primary" id="processBtn" style="max-width: 300px; padding: 15px 30px; font-size: 1.1em;">
                        🚀 Analyze & Estimate Navigation
                    </button>
                    <button class="btn-secondary" id="backBtn" style="max-width: 300px; padding: 12px 30px; margin-top: 10px; font-size: 1em;">← Back to Mode Selection</button>
                </div>
                
                <!-- Results Section -->
                <div id="resultsSection"></div>
            </div>
        `;
        
        this.setupVisuallyImpairedModeListeners();
    }
    
    setupBlindModeListeners() {
        // Image source toggle
        const cameraBtn = document.getElementById('cameraBtn');
        const uploadBox = document.getElementById('uploadBox');
        const imageInput = document.getElementById('imageInput');
        const cameraFeed = document.getElementById('cameraFeed');
        
        // Camera button
        cameraBtn.addEventListener('click', () => {
            this.blindModeUsingCamera = true;
            this.uploadedFile = null;
            this.uploadedFilename = null;
            document.getElementById('previewContainer').innerHTML = '';
            
            // Show camera feed
            cameraFeed.style.display = 'block';
            
            cameraBtn.style.background = '#28a745';
            cameraBtn.innerHTML = '✓ Camera Selected';
            
            // Start camera
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    this.videoStream = stream;
                    cameraFeed.srcObject = stream;
                })
                .catch(err => {
                    alert('Camera access denied: ' + err.message);
                    cameraBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                    cameraBtn.innerHTML = '📷 Use Camera';
                    cameraFeed.style.display = 'none';
                });
        });
        
        // Image upload for blind mode
        uploadBox.addEventListener('click', () => imageInput.click());
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('active');
        });
        uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('active'));
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('active');
            const files = e.dataTransfer.files;
            if (files.length > 0) this.handleBlindModeImageUpload(files[0]);
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleBlindModeImageUpload(e.target.files[0]);
            }
        });
        
        // Recording
        document.getElementById('recordBtn').addEventListener('click', () => this.toggleRecording());
        
        // Process
        document.getElementById('processBtn').addEventListener('click', () => this.processBlindMode());
        
        // Back button
        document.getElementById('backBtn').addEventListener('click', () => {
            // Stop camera if running
            if (this.videoStream) {
                this.videoStream.getTracks().forEach(track => track.stop());
                this.videoStream = null;
            }
            this.init();
        });
    }
    
    handleBlindModeImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', file);
        
        this.showLoading('Uploading image...');
        
        fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.uploadedFile = file;
                this.uploadedFilename = data.filename;
                this.blindModeUsingCamera = false;
                
                // Stop camera if running
                if (this.videoStream) {
                    this.videoStream.getTracks().forEach(track => track.stop());
                    this.videoStream = null;
                }
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('previewContainer').innerHTML = 
                        `<img src="${e.target.result}" class="preview-image">`;
                };
                reader.readAsDataURL(file);
                
                // Update button state
                const cameraBtn = document.getElementById('cameraBtn');
                cameraBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                cameraBtn.innerHTML = '📷 Use Camera';
                
                this.clearResults();
            } else {
                alert('Upload failed: ' + data.error);
            }
        })
        .catch(err => alert('Error: ' + err.message));
    }
    
    setupVisuallyImpairedModeListeners() {
        // Image source toggle
        const cameraBtn = document.getElementById('cameraBtn');
        const uploadBox = document.getElementById('uploadBox');
        const imageInput = document.getElementById('imageInput');
        const cameraFeed = document.getElementById('cameraFeed');
        
        // Camera button
        cameraBtn.addEventListener('click', () => {
            this.uploadedFile = null;
            this.uploadedFilename = null;
            document.getElementById('previewContainer').innerHTML = '';
            
            // Show camera feed
            cameraFeed.style.display = 'block';
            
            cameraBtn.style.background = '#28a745';
            cameraBtn.innerHTML = '✓ Camera Selected';
            
            // Start camera
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    this.videoStream = stream;
                    cameraFeed.srcObject = stream;
                })
                .catch(err => {
                    alert('Camera access denied: ' + err.message);
                    cameraBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                    cameraBtn.innerHTML = '📷 Use Camera';
                    cameraFeed.style.display = 'none';
                });
        });
        
        // Image upload
        uploadBox.addEventListener('click', () => imageInput.click());
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('active');
        });
        uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('active'));
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('active');
            const files = e.dataTransfer.files;
            if (files.length > 0) this.handleImageUploadVisuallyImpaired(files[0]);
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUploadVisuallyImpaired(e.target.files[0]);
            }
        });
        
        // Recording
        document.getElementById('recordBtn').addEventListener('click', () => this.toggleRecording());
        
        // Process
        document.getElementById('processBtn').addEventListener('click', () => this.processVisuallyImpairedMode());
        
        // Back button
        document.getElementById('backBtn').addEventListener('click', () => {
            // Stop camera if running
            if (this.videoStream) {
                this.videoStream.getTracks().forEach(track => track.stop());
                this.videoStream = null;
            }
            this.init();
        });
    }
    
    handleImageUploadVisuallyImpaired(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', file);
        
        this.showLoading('Uploading image...');
        
        fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.uploadedFile = file;
                this.uploadedFilename = data.filename;
                
                // Stop camera if running
                if (this.videoStream) {
                    this.videoStream.getTracks().forEach(track => track.stop());
                    this.videoStream = null;
                }
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('previewContainer').innerHTML = 
                        `<img src="${e.target.result}" class="preview-image">`;
                };
                reader.readAsDataURL(file);
                
                // Hide camera feed
                const cameraFeed = document.getElementById('cameraFeed');
                if (cameraFeed) {
                    cameraFeed.style.display = 'none';
                }
                
                // Update button state
                const cameraBtn = document.getElementById('cameraBtn');
                cameraBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                cameraBtn.innerHTML = '📷 Use Camera';
                
                this.clearResults();
            } else {
                alert('Upload failed: ' + data.error);
            }
        })
        .catch(err => alert('Error: ' + err.message));
    }
    
    initializeCamera() {
        // Only initialize if we're in blind mode and using camera
        if (this.mode !== 'blind' || !this.blindModeUsingCamera) {
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                this.videoStream = stream;
                const video = document.getElementById('cameraFeed');
                if (video) {
                    video.srcObject = stream;
                }
            })
            .catch(err => {
                console.error('Camera access denied:', err);
                if (this.mode === 'blind') {
                    this.showError('Camera access required. Please upload an image instead or enable camera access.');
                }
            });
    }
    
    handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', file);
        
        this.showLoading('Uploading image...');
        
        fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.uploadedFile = file;
                this.uploadedFilename = data.filename;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('previewContainer').innerHTML = 
                        `<img src="${e.target.result}" class="preview-image">`;
                };
                reader.readAsDataURL(file);
                
                this.clearResults();
            } else {
                alert('Upload failed: ' + data.error);
            }
        })
        .catch(err => alert('Error: ' + err.message));
    }
    
    toggleRecording() {
        if (!this.isRecording) {
            this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                this.mediaRecorder = new MediaRecorder(stream);
                this.audioChunks = [];
                
                this.mediaRecorder.ondataavailable = (event) => {
                    this.audioChunks.push(event.data);
                };
                
                this.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                    this.processAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.mediaRecorder.start();
                this.isRecording = true;
                document.getElementById('recordBtn').innerHTML = '⏹️ Stop Recording';
                document.getElementById('recordBtn').style.background = '#dc3545';
            })
            .catch(err => alert('Microphone access denied: ' + err.message));
    }
    
    stopRecording() {
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            const recordBtn = document.getElementById('recordBtn');
            if (recordBtn) {
                recordBtn.innerHTML = '🎙️ Start Recording';
                recordBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }
        }
    }
    
    processAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');
        
        this.showLoading('Transcribing audio...');
        
        fetch(`${API_BASE_URL}/transcribe`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (this.mode === 'blind') {
                    const processBtn = document.getElementById('processBtn');
                    if (processBtn) {
                        processBtn.style.display = 'block';
                    }
                } else {
                    document.getElementById('targetInput').value = data.target;
                }
                
                const transcript = `
                    <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <p style="color: #333; margin-bottom: 5px;"><strong>Transcribed:</strong> ${data.transcribed_text}</p>
                        <p style="color: #667eea;"><strong>Target:</strong> <strong style="font-size: 1.1em; color: #28a745;">${data.target}</strong></p>
                    </div>
                `;
                document.getElementById('audioTranscript').innerHTML = transcript;
                this.clearResults();
                
                // Auto-trigger processing after a short delay to allow UI to update
                setTimeout(() => {
                    if (this.mode === 'blind') {
                        this.processBlindMode();
                    } else {
                        this.processVisuallyImpairedMode();
                    }
                }, 500);
            } else {
                alert('Transcription failed: ' + data.error);
            }
        })
        .catch(err => alert('Error: ' + err.message));
    }
    
    processBlindMode() {
        const transcript = document.querySelector('#audioTranscript p:last-child');
        if (!transcript) {
            alert('Please record your request first');
            return;
        }
        
        const target = transcript.textContent.replace('Target: ', '').trim();
        
        if (this.blindModeUsingCamera) {
            // Capture from camera
            const video = document.getElementById('cameraFeed');
            if (!video || !video.videoWidth) {
                alert('Camera feed not ready');
                return;
            }
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');
                
                this.showLoading('Analyzing camera feed...');
                
                fetch(`${API_BASE_URL}/upload`, {
                    method: 'POST',
                    body: formData
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        this.processImageWithTarget(data.filename, target);
                    }
                });
            }, 'image/jpeg');
        } else {
            // Use uploaded image
            if (!this.uploadedFilename) {
                alert('Please upload an image first');
                return;
            }
            this.processImageWithTarget(this.uploadedFilename, target);
        }
    }
    
    processVisuallyImpairedMode() {
        const target = document.getElementById('targetInput').value.trim();
        if (!target) {
            alert('Please specify a target object');
            return;
        }
        
        const cameraFeed = document.getElementById('cameraFeed');
        const isCameraActive = cameraFeed && cameraFeed.style.display !== 'none' && cameraFeed.videoWidth > 0;
        
        if (isCameraActive) {
            // Capture from camera
            const canvas = document.createElement('canvas');
            canvas.width = cameraFeed.videoWidth;
            canvas.height = cameraFeed.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(cameraFeed, 0, 0);
            
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');
                
                this.showLoading('Analyzing camera feed...');
                
                fetch(`${API_BASE_URL}/upload`, {
                    method: 'POST',
                    body: formData
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        this.processImageWithTarget(data.filename, target);
                    }
                });
            }, 'image/jpeg');
        } else {
            // Use uploaded image
            if (!this.uploadedFilename) {
                alert('Please upload an image or use the camera first');
                return;
            }
            this.processImageWithTarget(this.uploadedFilename, target);
        }
    }
    
    processImageWithTarget(filename, target) {
        this.showLoading('Processing image...');
        
        fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                target: target
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.displayResults(data);
                this.generateInstruction(data.target, data.steps, data.angle);
            } else {
                this.showError(data.error);
            }
        })
        .catch(err => this.showError(err.message));
    }
    
    generateInstruction(target, steps, angle) {
        fetch(`${API_BASE_URL}/generate-instruction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: target,
                steps: steps,
                angle: angle
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.displayInstruction(data);
                // Auto-play the audio
                this.autoPlayAudio(data.audio_base64);
            }
        })
        .catch(err => console.error(err));
    }
    
    autoPlayAudio(audioBase64) {
        const audio = new Audio();
        audio.src = `data:audio/wav;base64,${audioBase64}`;
        audio.play().catch(err => {
            console.log('Audio auto-play was prevented by browser', err);
        });
    }
    
    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        
        const angleSign = data.angle > 0 ? '→' : '←';
        const directionEmoji = data.angle > 0 ? '➡️' : '⬅️';
        
        // Clear navigation instructions with full understanding
        const visual = `
            <div class="results">
                <h2 style="color: #667eea; margin-bottom: 40px; font-size: 1.8em; text-align: center;">🧭 Navigation Instructions</h2>
                
                <div style="margin-bottom: 30px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; text-align: center;">
                    <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Target Found: <strong>${data.target.toUpperCase()}</strong></p>
                    <p style="margin: 15px 0 0 0; font-size: 1.1em; opacity: 0.95;">Located approximately <strong>${data.distance_meters.toFixed(1)} meters</strong> away</p>
                </div>
                
                <div class="result-item" style="font-size: 1.3em; margin-bottom: 30px; padding: 25px; background: #fff3e0; border-radius: 12px; border-left: 6px solid #ff9800;">
                    <span class="result-label" style="color: #ff6b6b; font-weight: 600;">Step 1: Direction</span>
                    <span class="result-value" style="font-size: 1.5em; color: #ff6b6b; display: block; margin-top: 10px;">${directionEmoji} Turn ${Math.abs(data.angle).toFixed(1)}° to the ${data.angle > 0 ? 'RIGHT' : 'LEFT'}</span>
                </div>
                
                <div class="result-item" style="font-size: 1.3em; margin-bottom: 30px; padding: 25px; background: #e8f5e9; border-radius: 12px; border-left: 6px solid #28a745;">
                    <span class="result-label" style="color: #28a745; font-weight: 600;">Step 2: Walk</span>
                    <span class="result-value" style="font-size: 1.5em; color: #28a745; display: block; margin-top: 10px;">👟 Walk <strong>${Math.round(data.steps)} steps</strong> forward</span>
                </div>
                
                <div style="margin-bottom: 30px; padding: 20px; background: #f0f4ff; border-radius: 8px;">
                    <p style="margin: 0; font-size: 0.95em; color: #666; line-height: 1.8;">
                        <strong>📍 Summary:</strong> The <strong>${data.target.toUpperCase()}</strong> is <strong>${Math.round(data.steps)} steps</strong> away in the <strong>${data.angle > 0 ? 'right' : 'left'}</strong> direction (${Math.abs(data.angle).toFixed(1)}°). 
                        Walk ${Math.round(data.steps)} steps forward while turning to your ${data.angle > 0 ? 'right' : 'left'}.
                    </p>
                </div>
                
                ${this.mode === 'visually-impaired' ? `
                <div style="margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px;">
                    <h3 style="color: #333; margin-bottom: 15px; font-weight: 600; font-size: 1.1em;">📊 Detection Details</h3>
                    <div style="font-size: 0.95em; color: #666; line-height: 1.8;">
                        <p style="margin: 5px 0;">🎯 Target: <strong>${data.target.toUpperCase()}</strong></p>
                        <p style="margin: 5px 0;">📏 Distance: <strong>${data.distance_meters.toFixed(2)} meters</strong></p>
                        <p style="margin: 5px 0;">🎯 Confidence: <strong>${(data.confidence * 100).toFixed(0)}%</strong></p>
                        <p style="margin: 5px 0;">⚡ Processing time: <strong>${data.processing_time.toFixed(2)}s</strong></p>
                    </div>
                    <h3 style="color: #333; margin-top: 20px; margin-bottom: 15px; font-weight: 600; font-size: 1.1em;">🖼️ Visual Analysis</h3>
                    <img src="data:image/png;base64,${data.visualization}" alt="Visualization" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                </div>
                ` : `
                <div style="margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px;">
                    <h3 style="color: #333; margin-bottom: 15px; font-weight: 600; font-size: 1.1em;">📊 Detection Details</h3>
                    <div style="font-size: 0.95em; color: #666; line-height: 1.8;">
                        <p style="margin: 5px 0;">🎯 Target: <strong>${data.target.toUpperCase()}</strong></p>
                        <p style="margin: 5px 0;">📏 Distance: <strong>${data.distance_meters.toFixed(2)} meters</strong></p>
                        <p style="margin: 5px 0;">🎯 Confidence: <strong>${(data.confidence * 100).toFixed(0)}%</strong></p>
                        <p style="margin: 5px 0;">⚡ Processing time: <strong>${data.processing_time.toFixed(2)}s</strong></p>
                    </div>
                </div>
                `}
                
                <div id="instructionContainer"></div>
            </div>
        `;
        
        resultsSection.innerHTML = visual;
    }
    
    displayInstruction(data) {
        const container = document.getElementById('instructionContainer');
        if (container) {
            const instructionHTML = `
                <div style="margin-top: 40px; padding: 25px; background: #fff3cd; border-radius: 12px; border-left: 6px solid #ff9800;">
                    <h3 style="color: #ff9800; margin-top: 0; margin-bottom: 15px; font-size: 1.2em;">🔊 Voice Instruction</h3>
                    <p style="margin: 0; font-size: 1.1em; line-height: 1.8; color: #333; font-weight: 500;">${data.conversational_text}</p>
                    ${data.audio_base64 ? `
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ffc107;">
                        <audio controls style="width: 100%; margin-top: 10px;">
                            <source src="data:audio/wav;base64,${data.audio_base64}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                        <p style="font-size: 0.85em; color: #666; margin-top: 10px;">Click play to hear the instruction read aloud</p>
                    </div>
                    ` : ''}
                </div>
            `;
            container.innerHTML = instructionHTML;
        }
    }
    
    showLoading(message) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                ${message}
            </div>
        `;
    }
    
    showError(message) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.innerHTML = `<div class="error-box">❌ Error: ${message}</div>`;
    }
    
    clearResults() {
        document.getElementById('resultsSection').innerHTML = '';
    }
    
    checkBackendHealth() {
        fetch(`${API_BASE_URL.replace('/api', '')}/health`)
            .then(res => res.json())
            .catch(() => {
                console.warn('Backend is not running. Start it with: python3 main.py on port 5001');
            });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new NavigationApp();
});

