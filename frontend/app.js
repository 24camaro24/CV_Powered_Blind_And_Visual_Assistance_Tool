const API_BASE_URL = '/api';

class NavigationApp {
    constructor() {
        this.uploadedFilename = null;
        this.supportFiles = [];
        this.isUsingCamera = false;
        this.isRecording = false;
        this.isProcessing = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.videoStream = null;
        this.init();
    }

    init() {
        this.renderApp();
        this.bindEvents();
        this.checkBackendHealth();
        window.addEventListener('beforeunload', () => this.stopCamera());
    }

    renderApp() {
        document.getElementById('root').innerHTML = `
            <main class="container">
                <header class="app-header">
                    <h1>Navigation Assistant</h1>
                </header>

                <div class="grid">
                    <section class="section" aria-labelledby="imageSourceTitle">
                        <h2 id="imageSourceTitle">Image Source</h2>

                        <div class="source-actions">
                            <button class="btn-primary" id="cameraBtn" type="button">Use Camera</button>
                            <button class="btn-secondary" id="clearSourceBtn" type="button">Clear Image</button>
                        </div>

                        <video id="cameraFeed" class="camera-preview" autoplay playsinline muted></video>

                        <div class="upload-box" id="uploadBox" role="button" tabindex="0" aria-label="Upload image">
                            <p class="upload-text">Drop image or browse</p>
                            <p class="upload-hint">JPG, PNG, GIF, BMP up to 50 MB</p>
                            <input type="file" id="imageInput" accept="image/*">
                            <div id="previewContainer"></div>
                        </div>

                        <p id="sourceStatus" class="status-line">No image selected</p>
                    </section>

                    <section class="section" aria-labelledby="targetTitle">
                        <h2 id="targetTitle">Target</h2>

                        <label class="field-label" for="targetInput">Object to find</label>
                        <input type="text" id="targetInput" class="target-input" placeholder="keys, door, chair">

                        <div class="source-actions">
                            <button class="btn-primary" id="recordBtn" type="button">Start Recording</button>
                            <button class="btn-secondary" id="clearTargetBtn" type="button">Clear Target</button>
                        </div>

                        <div id="audioTranscript" class="transcript-box hidden"></div>
                    </section>
                </div>

                <div class="actions">
                    <button class="btn-primary primary-action" id="processBtn" type="button">Get Guidance</button>
                </div>

                <section class="section example-section" aria-labelledby="exampleTitle">
                    <h2 id="exampleTitle">Find by Example</h2>

                    <label class="field-label" for="exampleLabelInput">Object label</label>
                    <input type="text" id="exampleLabelInput" class="target-input" placeholder="my keys">

                    <div class="upload-box" id="supportUploadBox" role="button" tabindex="0" aria-label="Upload example images">
                        <p class="upload-text">Drop example images or browse</p>
                        <p class="upload-hint">Up to 10 object images</p>
                        <input type="file" id="supportInput" accept="image/*" multiple>
                        <div id="supportPreviewContainer" class="support-preview-grid"></div>
                    </div>

                    <p id="supportStatus" class="status-line">No example images selected</p>

                    <div class="source-actions">
                        <button class="btn-primary" id="exampleProcessBtn" type="button">Find by Example</button>
                        <button class="btn-secondary" id="clearSupportBtn" type="button">Clear Examples</button>
                    </div>
                </section>

                <div id="resultsSection" aria-live="polite"></div>
            </main>
        `;
    }

    bindEvents() {
        const cameraBtn = this.el('cameraBtn');
        const clearSourceBtn = this.el('clearSourceBtn');
        const uploadBox = this.el('uploadBox');
        const imageInput = this.el('imageInput');
        const recordBtn = this.el('recordBtn');
        const clearTargetBtn = this.el('clearTargetBtn');
        const processBtn = this.el('processBtn');
        const targetInput = this.el('targetInput');
        const supportUploadBox = this.el('supportUploadBox');
        const supportInput = this.el('supportInput');
        const clearSupportBtn = this.el('clearSupportBtn');
        const exampleProcessBtn = this.el('exampleProcessBtn');

        cameraBtn.addEventListener('click', () => this.toggleCamera());
        clearSourceBtn.addEventListener('click', () => this.clearSource());
        recordBtn.addEventListener('click', () => this.toggleRecording());
        clearTargetBtn.addEventListener('click', () => this.clearTarget());
        processBtn.addEventListener('click', () => this.processNavigation());
        targetInput.addEventListener('input', () => this.clearResults());
        clearSupportBtn.addEventListener('click', () => this.clearSupportExamples());
        exampleProcessBtn.addEventListener('click', () => this.processExampleNavigation());

        uploadBox.addEventListener('click', () => imageInput.click());
        uploadBox.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                imageInput.click();
            }
        });

        uploadBox.addEventListener('dragover', (event) => {
            event.preventDefault();
            uploadBox.classList.add('active');
        });

        uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('active'));

        uploadBox.addEventListener('drop', (event) => {
            event.preventDefault();
            uploadBox.classList.remove('active');
            const file = event.dataTransfer.files[0];
            if (file) {
                this.handleImageUpload(file);
            }
        });

        imageInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                this.handleImageUpload(file);
            }
        });

        supportUploadBox.addEventListener('click', () => supportInput.click());
        supportUploadBox.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                supportInput.click();
            }
        });

        supportUploadBox.addEventListener('dragover', (event) => {
            event.preventDefault();
            supportUploadBox.classList.add('active');
        });

        supportUploadBox.addEventListener('dragleave', () => supportUploadBox.classList.remove('active'));

        supportUploadBox.addEventListener('drop', (event) => {
            event.preventDefault();
            supportUploadBox.classList.remove('active');
            this.handleSupportUpload(event.dataTransfer.files);
        });

        supportInput.addEventListener('change', (event) => {
            this.handleSupportUpload(event.target.files);
        });
    }

    el(id) {
        return document.getElementById(id);
    }

    async toggleCamera() {
        if (this.isUsingCamera) {
            this.stopCamera();
            this.setSourceStatus('No image selected');
            return;
        }

        await this.startCamera();
    }

    async startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Camera access is not available in this browser.');
            return;
        }

        try {
            this.stopCamera();

            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' }
                },
                audio: false
            };

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (_) {
                stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }

            this.videoStream = stream;
            this.isUsingCamera = true;
            this.uploadedFilename = null;

            const video = this.el('cameraFeed');
            video.srcObject = stream;
            video.classList.add('active');
            await video.play().catch(() => {});

            this.el('previewContainer').innerHTML = '';
            this.el('imageInput').value = '';
            this.el('cameraBtn').textContent = 'Stop Camera';
            this.el('cameraBtn').classList.add('is-active');
            this.setSourceStatus('Camera is active');
            this.clearResults();
        } catch (error) {
            this.stopCamera();
            this.showError(`Camera access failed: ${error.message}`);
        }
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach((track) => track.stop());
        }

        this.videoStream = null;
        this.isUsingCamera = false;

        const video = this.el('cameraFeed');
        if (video) {
            video.pause();
            video.srcObject = null;
            video.classList.remove('active');
        }

        const cameraBtn = this.el('cameraBtn');
        if (cameraBtn) {
            cameraBtn.textContent = 'Use Camera';
            cameraBtn.classList.remove('is-active');
        }
    }

    async handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            this.showError('Please choose an image file.');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            this.showError('Image is too large. Maximum size is 50 MB.');
            return;
        }

        try {
            this.stopCamera();
            const data = await this.uploadImage(file, file.name, 'Uploading image...');

            this.uploadedFilename = data.filename;
            this.renderPreview(data.image_base64, file.name, file.type);
            this.setSourceStatus(`Image selected: ${file.name}`);
            this.clearResults();
        } catch (error) {
            this.showError(error.message);
        }
    }

    async uploadImage(blob, filename, loadingMessage) {
        const formData = new FormData();
        formData.append('image', blob, filename);

        this.showLoading(loadingMessage);
        const data = await this.requestJSON(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!data.success || !data.filename) {
            throw new Error(data.error || 'Image upload failed.');
        }

        return data;
    }

    renderPreview(imageBase64, name, mimeType = 'image/png') {
        const preview = imageBase64
            ? `<img class="preview-image" src="data:${this.escapeHTML(mimeType)};base64,${imageBase64}" alt="${this.escapeHTML(name)}">`
            : '';

        this.el('previewContainer').innerHTML = preview;
    }

    clearSource() {
        this.stopCamera();
        this.uploadedFilename = null;
        this.el('imageInput').value = '';
        this.el('previewContainer').innerHTML = '';
        this.setSourceStatus('No image selected');
        this.clearResults();
    }

    handleSupportUpload(fileList) {
        const files = Array.from(fileList || []);
        if (files.length === 0) {
            return;
        }

        try {
            const nextFiles = [...this.supportFiles];
            for (const file of files) {
                this.validateImageFile(file);
                nextFiles.push(file);
            }

            if (nextFiles.length > 10) {
                throw new Error('At most 10 example images are supported.');
            }

            this.supportFiles = nextFiles;
            this.el('supportInput').value = '';
            this.renderSupportPreviews();
            this.setSupportStatus(`${this.supportFiles.length} example image${this.supportFiles.length === 1 ? '' : 's'} selected`);
            this.clearResults();
        } catch (error) {
            this.showError(error.message);
        }
    }

    validateImageFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            throw new Error('Please choose image files only.');
        }

        if (file.size > 50 * 1024 * 1024) {
            throw new Error('Example image is too large. Maximum size is 50 MB.');
        }
    }

    renderSupportPreviews() {
        const container = this.el('supportPreviewContainer');
        if (!container) {
            return;
        }

        if (this.supportFiles.length === 0) {
            container.innerHTML = '';
            return;
        }

        Promise.all(this.supportFiles.map((file, index) => this.readFileAsDataURL(file).then((src) => ({
            src,
            index,
            name: file.name
        })))).then((items) => {
            container.innerHTML = items.map((item) => `
                <div class="support-preview-item">
                    <img src="${item.src}" alt="${this.escapeHTML(item.name)}">
                    <span>${item.index + 1}</span>
                </div>
            `).join('');
        });
    }

    readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.addEventListener('load', () => resolve(reader.result));
            reader.addEventListener('error', () => reject(new Error(`Could not read ${file.name}`)));
            reader.readAsDataURL(file);
        });
    }

    clearSupportExamples() {
        this.supportFiles = [];
        this.el('supportInput').value = '';
        this.el('supportPreviewContainer').innerHTML = '';
        this.setSupportStatus('No example images selected');
        this.clearResults();
    }

    setSupportStatus(message) {
        const status = this.el('supportStatus');
        if (status) {
            status.textContent = message;
        }
    }

    setSourceStatus(message) {
        const status = this.el('sourceStatus');
        if (status) {
            status.textContent = message;
        }
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Microphone access is not available in this browser.');
            return;
        }

        if (!window.MediaRecorder) {
            this.showError('Audio recording is not supported in this browser.');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const options = this.getRecordingOptions();

            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(stream, options);

            this.mediaRecorder.addEventListener('dataavailable', (event) => {
                if (event.data && event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            });

            this.mediaRecorder.addEventListener('stop', () => {
                stream.getTracks().forEach((track) => track.stop());
                const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                this.isRecording = false;
                this.updateRecordingUI(false);
                this.processAudio(audioBlob, this.audioExtensionFor(mimeType));
            });

            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateRecordingUI(true);
            this.el('audioTranscript').classList.add('hidden');
            this.el('audioTranscript').innerHTML = '';
            this.clearResults();
        } catch (error) {
            this.isRecording = false;
            this.updateRecordingUI(false);
            this.showError(`Microphone access failed: ${error.message}`);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
    }

    getRecordingOptions() {
        const candidates = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4'
        ];

        const mimeType = candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
        return mimeType ? { mimeType } : {};
    }

    audioExtensionFor(mimeType) {
        if (mimeType.includes('mp4')) {
            return 'm4a';
        }
        if (mimeType.includes('wav')) {
            return 'wav';
        }
        return 'webm';
    }

    updateRecordingUI(isRecording) {
        const recordBtn = this.el('recordBtn');
        if (!recordBtn) {
            return;
        }

        recordBtn.textContent = isRecording ? 'Stop Recording' : 'Start Recording';
        recordBtn.classList.toggle('recording', isRecording);
    }

    async processAudio(audioBlob, extension) {
        if (!audioBlob || audioBlob.size === 0) {
            this.showError('No audio was recorded.');
            return;
        }

        const formData = new FormData();
        formData.append('audio', audioBlob, `request.${extension}`);

        try {
            this.showLoading('Transcribing audio...');
            const data = await this.requestJSON(`${API_BASE_URL}/transcribe`, {
                method: 'POST',
                body: formData
            });

            if (!data.success) {
                throw new Error(data.error || 'Transcription failed.');
            }

            const target = data.target || data.transcribed_text || '';
            this.el('targetInput').value = target;

            const transcript = this.el('audioTranscript');
            transcript.classList.remove('hidden');
            transcript.innerHTML = `
                <p><strong>Transcript:</strong> ${this.escapeHTML(data.transcribed_text || '')}</p>
                <p><strong>Target:</strong> ${this.escapeHTML(target)}</p>
            `;

            this.clearResults();
        } catch (error) {
            this.showError(error.message);
        }
    }

    clearTarget() {
        this.el('targetInput').value = '';
        this.el('audioTranscript').innerHTML = '';
        this.el('audioTranscript').classList.add('hidden');
        this.clearResults();
    }

    async processNavigation() {
        if (this.isProcessing) {
            return;
        }

        const target = this.el('targetInput').value.trim();
        if (!target) {
            this.showError('Enter or record a target object.');
            return;
        }

        try {
            this.setProcessing(true);
            let filename = this.uploadedFilename;

            if (this.isUsingCamera) {
                const frame = await this.captureCameraFrame();
                const upload = await this.uploadImage(frame, 'camera-frame.jpg', 'Uploading camera frame...');
                filename = upload.filename;
            }

            if (!filename) {
                throw new Error('Choose an image or turn on the camera first.');
            }

            await this.processImageWithTarget(filename, target);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setProcessing(false);
        }
    }

    async processExampleNavigation() {
        if (this.isProcessing) {
            return;
        }

        if (this.supportFiles.length === 0) {
            this.showError('Add at least one example image.');
            return;
        }

        try {
            this.setProcessing(true);
            let filename = this.uploadedFilename;

            if (this.isUsingCamera) {
                const frame = await this.captureCameraFrame();
                const upload = await this.uploadImage(frame, 'camera-frame.jpg', 'Uploading camera frame...');
                filename = upload.filename;
            }

            if (!filename) {
                throw new Error('Choose a scene image or turn on the camera first.');
            }

            const label = this.el('exampleLabelInput').value.trim() || 'example object';
            const formData = new FormData();
            formData.append('filename', filename);
            formData.append('target', label);
            this.supportFiles.forEach((file) => {
                formData.append('support_images', file, file.name);
            });

            this.showLoading('Finding by example...');
            const data = await this.requestJSON(`${API_BASE_URL}/process-example`, {
                method: 'POST',
                body: formData
            });

            this.displayResults(data);
            this.generateInstruction(data);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setProcessing(false);
        }
    }

    captureCameraFrame() {
        const video = this.el('cameraFeed');

        if (!this.isUsingCamera || !video || !video.videoWidth || !video.videoHeight) {
            throw new Error('Camera is not ready yet.');
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0);

        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Could not capture camera frame.'));
                }
            }, 'image/jpeg', 0.92);
        });
    }

    async processImageWithTarget(filename, target) {
        this.showLoading('Processing image...');

        const data = await this.requestJSON(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, target })
        });

        if (!data.success) {
            throw new Error(data.error || 'Image processing failed.');
        }

        this.displayResults(data);
        this.generateInstruction(data);
    }

    async generateInstruction(result) {
        try {
            const data = await this.requestJSON(`${API_BASE_URL}/generate-instruction`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target: result.target,
                    steps: result.steps,
                    angle: result.angle,
                    distance_meters: result.distance_meters,
                    confidence: result.confidence,
                    depth: result.depth,
                    surfaces: result.surfaces || []
                })
            });

            if (data.success) {
                this.displayInstruction(data);
                this.autoPlayAudio(data.audio_base64);
            }
        } catch (error) {
            console.warn('Instruction generation failed:', error);
        }
    }

    displayResults(data) {
        const angle = this.asNumber(data.angle, 0);
        const steps = this.asNumber(data.steps, 0);
        const distance = this.asNumber(data.distance_meters, null);
        const confidence = this.asNumber(data.confidence, null);
        const processingTime = this.asNumber(data.processing_time, null);
        const direction = this.directionFor(angle);
        const target = this.escapeHTML(data.target || 'target');
        const guidanceText = data.navigation_guidance && data.navigation_guidance.conversational_text
            ? data.navigation_guidance.conversational_text
            : '';

        this.el('resultsSection').innerHTML = `
            <section class="results">
                <h2>Guidance</h2>

                <div class="guidance-summary">
                    <p class="summary-target">${target}</p>
                    <p class="summary-action">${this.escapeHTML(direction.action)}</p>
                    <p class="summary-detail">${this.escapeHTML(direction.detail)}</p>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <span class="metric-label">Distance</span>
                        <strong class="metric-value">${this.formatMeters(distance)}</strong>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Steps</span>
                        <strong class="metric-value">${Math.max(0, Math.round(steps))}</strong>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Angle</span>
                        <strong class="metric-value">${Math.abs(angle).toFixed(1)} deg</strong>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Confidence</span>
                        <strong class="metric-value">${this.formatPercent(confidence)}</strong>
                    </div>
                </div>

                <div class="result-item">
                    <span class="result-label">Surface</span>
                    <span class="result-value">${this.formatSurfaces(data.surfaces)}</span>
                </div>

                <div class="result-item">
                    <span class="result-label">Processing time</span>
                    <span class="result-value">${this.formatSeconds(processingTime)}</span>
                </div>

                ${data.few_shot ? `
                    <div class="result-item">
                        <span class="result-label">Example match</span>
                        <span class="result-value">${this.formatFewShot(data.few_shot)}</span>
                    </div>
                ` : ''}

                ${data.visualization ? `
                    <div class="visualization">
                        <h3>Visual Analysis</h3>
                        <img src="data:image/png;base64,${data.visualization}" alt="Detected target visualization">
                    </div>
                ` : ''}

                <div id="instructionContainer">
                    ${guidanceText ? this.renderInstructionBox({ conversational_text: guidanceText }) : ''}
                </div>
            </section>
        `;
    }

    displayInstruction(data) {
        const container = this.el('instructionContainer');
        if (!container) {
            return;
        }

        container.innerHTML = this.renderInstructionBox(data);
    }

    renderInstructionBox(data) {
        const text = data.conversational_text || data.detailed_text || '';
        const audio = data.audio_base64
            ? `
                <audio class="audio-player" controls>
                    <source src="data:audio/wav;base64,${data.audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            `
            : '';

        if (!text && !audio) {
            return '';
        }

        return `
            <div class="instruction-box">
                <h3>Voice Instruction</h3>
                ${text ? `<p>${this.escapeHTML(text)}</p>` : ''}
                ${audio}
            </div>
        `;
    }

    directionFor(angle) {
        const absAngle = Math.abs(angle);

        if (absAngle <= 2) {
            return {
                action: 'Go straight',
                detail: 'The target is close to the center of the frame.'
            };
        }

        const side = angle > 0 ? 'right' : 'left';
        return {
            action: `Turn ${side}`,
            detail: `Rotate ${absAngle.toFixed(1)} degrees to the ${side}, then walk forward.`
        };
    }

    async requestJSON(url, options) {
        const response = await fetch(url, options);
        let data = {};

        try {
            data = await response.json();
        } catch (_) {
            data = {};
        }

        if (!response.ok || data.error || data.success === false) {
            throw new Error(data.error || `Request failed with status ${response.status}`);
        }

        return data;
    }

    autoPlayAudio(audioBase64) {
        if (!audioBase64) {
            return;
        }

        const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
        audio.play().catch((error) => {
            console.log('Audio auto-play was prevented by the browser.', error);
        });
    }

    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
        [
            ['processBtn', 'Get Guidance'],
            ['exampleProcessBtn', 'Find by Example']
        ].forEach(([id, idleText]) => {
            const button = this.el(id);
            if (button) {
                button.disabled = isProcessing;
                button.textContent = isProcessing ? 'Working...' : idleText;
            }
        });
    }

    showLoading(message) {
        this.el('resultsSection').innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <span>${this.escapeHTML(message)}</span>
            </div>
        `;
    }

    showError(message) {
        this.el('resultsSection').innerHTML = `
            <div class="error-box">
                <strong>Error:</strong> ${this.escapeHTML(message)}
            </div>
        `;
    }

    clearResults() {
        const results = this.el('resultsSection');
        if (results) {
            results.innerHTML = '';
        }
    }

    formatMeters(value) {
        return Number.isFinite(value) ? `${value.toFixed(1)} m` : 'Unknown';
    }

    formatPercent(value) {
        return Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'Unknown';
    }

    formatSeconds(value) {
        return Number.isFinite(value) ? `${value.toFixed(2)} s` : 'Unknown';
    }

    formatSurfaces(surfaces) {
        if (!Array.isArray(surfaces) || surfaces.length === 0) {
            return 'Not detected';
        }

        return surfaces
            .map((surface) => {
                const name = this.escapeHTML(surface.surface || 'surface');
                const confidence = this.asNumber(surface.confidence, null);
                return Number.isFinite(confidence)
                    ? `${name} (${Math.round(confidence * 100)}%)`
                    : name;
            })
            .join(', ');
    }

    formatFewShot(fewShot) {
        const existence = this.asNumber(fewShot.existence_prob, null);
        const localizer = this.asNumber(fewShot.localizer_score, null);
        const bgProb = this.asNumber(fewShot.bg_prob, null);
        const parts = [];

        if (Number.isFinite(existence)) {
            parts.push(`similarity ${Math.round(existence * 100)}%`);
        }
        if (Number.isFinite(localizer)) {
            parts.push(`box ${Math.round(localizer * 100)}%`);
        }
        if (Number.isFinite(bgProb)) {
            parts.push(`background ${Math.round(bgProb * 100)}%`);
        }

        return parts.length > 0 ? parts.join(', ') : 'Available';
    }

    asNumber(value, fallback) {
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    }

    escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, (character) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[character]));
    }

    checkBackendHealth() {
        fetch('/health')
            .then((response) => response.json())
            .catch(() => {
                console.warn('Backend is not running. Start it with: python main.py');
            });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new NavigationApp();
});
