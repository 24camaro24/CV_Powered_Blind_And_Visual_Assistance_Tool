import os
# DISABLE CUDA FIRST - before torch is imported
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Disable HuggingFace model downloads if cache not available
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online but will fallback to cache
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import base64
from pathlib import Path
import signal
import sys

# Disable CUDA in torch explicitly after import
try:
    # Patch cuda module to fake CUDA support
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.current_device = lambda: -1
    torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"
    torch.cuda._initialized = False
    torch.cuda.is_initialized = lambda *args, **kwargs: False
    torch.cuda.init = lambda *args, **kwargs: None
    torch._C._cuda_init = None
    
    # Additional aggressive CUDA disabling
    torch.cuda.device = lambda x: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.synchronize = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
    torch.cuda.memory_stats = lambda *args, **kwargs: {}
    torch.cuda.mem_get_info = lambda *args, **kwargs: (1024*1024*1024, 1024*1024*1024)  # Fake 1GB free
    
    # CRITICAL FIX: Monkeypatch Tensor.to() to handle device parameter correctly
    original_tensor_to = torch.Tensor.to
    def fixed_tensor_to(self, *args, **kwargs):
        """Fixed .to() that handles bad device parameters"""
        try:
            # Try normal .to() first
            return original_tensor_to(self, *args, **kwargs)
        except TypeError as e:
            if "to() received an invalid combination" in str(e):
                # If .to() gets bad arguments, try to fix them
                # Check if device is being passed as dtype
                if len(args) > 0:
                    first_arg = args[0]
                    if isinstance(first_arg, torch.device):
                        # Device is a torch.device object - convert to CPU directly
                        return self.cpu()
                    elif isinstance(first_arg, str):
                        # Device is a string
                        if first_arg == 'cpu' or 'cpu' in str(first_arg):
                            return self.cpu()
                # If we can't figure it out, just return on CPU
                return self.cpu()
            else:
                raise
    
    torch.Tensor.to = fixed_tensor_to
    print("[CUDA PATCH] Monkeypatched Tensor.to() to handle device errors")
    
    # FIX: Patch subtract operation to handle bool tensors
    # When bool tensors are subtracted, convert to float first
    original_sub = torch.Tensor.__sub__
    def fixed_sub(self, other):
        if self.dtype == torch.bool:
            return original_sub(self.float(), other)
        elif isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            return original_sub(self, other.float())
        return original_sub(self, other)
    torch.Tensor.__sub__ = fixed_sub
    
    original_rsub = torch.Tensor.__rsub__
    def fixed_rsub(self, other):
        if self.dtype == torch.bool:
            return original_rsub(self.float(), other)
        elif isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            return original_rsub(self, other.float())
        return original_rsub(self, other)
    torch.Tensor.__rsub__ = fixed_rsub
    
    # Also patch torch.sub function
    original_torch_sub = torch.sub
    def fixed_torch_sub(input, other, *, alpha=1, out=None):
        if input.dtype == torch.bool:
            input = input.float()
        if isinstance(other, torch.Tensor) and other.dtype == torch.bool:
            other = other.float()
        return original_torch_sub(input, other, alpha=alpha, out=out)
    torch.sub = fixed_torch_sub
    
    # FIX: Patch torch.finfo() to handle device objects passed as dtype
    original_finfo = torch.finfo
    def fixed_finfo(dtype):
        """Handle case where torch.device is passed instead of dtype"""
        if isinstance(dtype, torch.device):
            # Device passed instead of dtype - return float32 info
            return original_finfo(torch.float32)
        return original_finfo(dtype)
    torch.finfo = fixed_finfo
    
    # FIX: Patch torch.iinfo() to handle device objects passed as dtype
    original_iinfo = torch.iinfo
    def fixed_iinfo(dtype):
        """Handle case where torch.device is passed instead of dtype"""
        if isinstance(dtype, torch.device):
            # Device passed instead of dtype - return int32 info
            return original_iinfo(torch.int32)
        return original_iinfo(dtype)
    torch.iinfo = fixed_iinfo
    
    print("[CUDA PATCH] Monkeypatched subtraction operations for bool tensors")
except Exception as e:
    print(f"[CUDA Patch] Warning: {e}")

# Ensure no MPS either (macOS specific)
if hasattr(torch.backends, 'mps'):
    try:
        torch.backends.mps.enabled = False
    except:
        pass

# FIX: Patch BertModel BEFORE importing GroundingDINO
try:
    from transformers.models.bert.modeling_bert import BertModel
    if not hasattr(BertModel, 'get_head_mask'):
        def get_head_mask(self, head_mask, num_hidden_layers):
            """Create a mask from the two representations of the head_mask."""
            if head_mask is not None:
                if head_mask.size()[0] != num_hidden_layers:
                    raise ValueError(
                        f"The head_mask should be specified for {num_hidden_layers} layers, but it was for"
                        f" {head_mask.size()[0]}."
                    )
                head_mask = head_mask.to(dtype=torch.float32)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        BertModel.get_head_mask = get_head_mask
        print("[PATCH] Added get_head_mask to BertModel")
except Exception as e:
    print(f"[PATCH WARNING] Could not patch BertModel: {e}")

# GroundingDINO
from groundingdino.util.inference import load_model, predict, load_image
import groundingdino.util.inference as groundingdino_inference

# FIX: Patch GroundingDINO's predict() to handle device properly
original_predict = groundingdino_inference.predict

def patched_predict(model, image, caption, box_threshold, text_threshold, device='cpu', remove_combined=False):
    """
    Fixed version - NO .to() calls. Subtraction operator patched for bool tensors.
    """
    from groundingdino.util.inference import preprocess_caption, get_phrases_from_posmap
    import bisect
    
    # Ensure image is float32
    if isinstance(image, torch.Tensor) and image.dtype != torch.float32:
        image = image.float()
    
    caption = preprocess_caption(caption=caption)
    
    with torch.no_grad():
        # Forward pass - bool tensor subtraction is now handled by patched __sub__
        outputs = model(image[None], captions=[caption])
    
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]
    
    # These comparisons will create bool tensors, but subtraction on them will work
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
    
    return boxes, logits.max(dim=1)[0] if len(logits) > 0 else torch.tensor([]), phrases

# Replace the predict function
groundingdino_inference.predict = patched_predict
predict = patched_predict
print("[PATCH] GroundingDINO predict() patched - Tensor.to() fix in place")


# Depth Anything V2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Audio LLM (faster-whisper)
from faster_whisper import WhisperModel

# Local instruction LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# TTS - Import lazily to avoid slow scipy load on macOS
TTS = None  # Will be imported on first use


# Timeout helper for long-running operations
def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Model loading operation timed out")


def load_with_timeout(load_func, timeout_secs=30, model_name="model"):
    """Wrap model loading with timeout protection"""
    if hasattr(signal, 'SIGALRM'):  # Unix-like systems only
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_secs)
            result = load_func()
            signal.alarm(0)  # Cancel alarm
            print(f"[SUCCESS] {model_name} loaded successfully")
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            print(f"[TIMEOUT] {model_name} loading took too long (>{timeout_secs}s), may be downloading...")
            return None
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            raise
    else:
        # Windows doesn't support SIGALRM, just try normally
        return load_func()


# Siamese Network for Few-Shot Object Matching
class SiameseNetwork(torch.nn.Module):
    """
    Siamese Network for few-shot object matching.
    Learns embeddings from reference images and matches objects in new scenes.
    """
    
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        
        # Lightweight ResNet-18 backbone for feature extraction
        from torchvision import models
        resnet18 = models.resnet18(pretrained=True)
        
        # Remove classification head
        self.backbone = torch.nn.Sequential(*list(resnet18.children())[:-1])
        
        # Add embedding projection layer
        self.embedding_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        self.device = 'cpu'
        
    def forward(self, x):
        """Extract embedding from image"""
        if x.dtype != torch.float32:
            x = x.float()
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.embedding_head(features)
        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.device = device if isinstance(device, str) else str(device)
        return self


class FewShotMatcher:
    """
    Manages few-shot learning for object detection using Siamese networks.
    Stores reference images and matches objects in new scenes.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.siamese = SiameseNetwork(embedding_dim=256).to(device)
        self.siamese.eval()
        
        # Reference database: {object_name: [embeddings, image_data]}
        self.reference_db = {}
        
        # Put model in eval mode
        for param in self.siamese.parameters():
            param.requires_grad = False
    
    def add_reference(self, object_name, image_tensor):
        """
        Add reference image for few-shot learning.
        
        Args:
            object_name: Name of the object (e.g., "my_phone", "speaker")
            image_tensor: Input image as tensor (C, H, W) or PIL Image or numpy array
        
        Returns:
            embedding: The computed embedding for this reference
        """
        # Convert to tensor if needed
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image_tensor, Image.Image):
            image_tensor = torch.from_numpy(np.array(image_tensor)).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image_tensor, torch.Tensor):
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
        
        # Normalize to [0, 1]
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Add batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Compute embedding
        with torch.no_grad():
            embedding = self.siamese(image_tensor.to(self.device))
        
        # Store in database
        if object_name not in self.reference_db:
            self.reference_db[object_name] = {
                'embeddings': [],
                'count': 0
            }
        
        self.reference_db[object_name]['embeddings'].append(embedding.cpu().detach())
        self.reference_db[object_name]['count'] += 1
        
        print(f"[FewShot] Added reference for '{object_name}' (count: {self.reference_db[object_name]['count']})")
        
        return embedding
    
    def match_in_region(self, image_region, similarity_threshold=0.65):
        """
        Match an image region against all stored references.
        
        Args:
            image_region: Image region to match (as tensor, PIL Image, or numpy array)
            similarity_threshold: Minimum similarity score to consider a match
        
        Returns:
            matched_objects: List of dicts with {object_name, similarity_score, embedding}
        """
        if len(self.reference_db) == 0:
            return []
        
        # Convert input to tensor
        if isinstance(image_region, np.ndarray):
            image_tensor = torch.from_numpy(image_region).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image_region, Image.Image):
            image_tensor = torch.from_numpy(np.array(image_region)).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image_region, torch.Tensor):
            image_tensor = image_region.float()
        else:
            return []
        
        # Normalize
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Add batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Compute embedding
        with torch.no_grad():
            query_embedding = self.siamese(image_tensor.to(self.device))
        
        # Compare against all references
        matched_objects = []
        for object_name, data in self.reference_db.items():
            embeddings = torch.cat(data['embeddings'], dim=0)  # (num_refs, embedding_dim)
            
            # Compute cosine similarity with all references
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding,  # (1, embedding_dim)
                embeddings        # (num_refs, embedding_dim)
            )  # -> (num_refs,)
            
            # Take max similarity (best match)
            max_sim = similarities.max().item()
            
            if max_sim >= similarity_threshold:
                matched_objects.append({
                    'object_name': object_name,
                    'similarity': float(max_sim),
                    'num_references': data['count']
                })
        
        # Sort by similarity score
        matched_objects = sorted(matched_objects, key=lambda x: x['similarity'], reverse=True)
        
        return matched_objects
    
    def get_best_match(self, image_region, similarity_threshold=0.65):
        """
        Get the single best match for an image region.
        
        Returns:
            best_match: Dict with {object_name, similarity, num_references} or None
        """
        matches = self.match_in_region(image_region, similarity_threshold)
        return matches[0] if matches else None
    
    def clear_references(self, object_name=None):
        """Clear reference images for a specific object or all objects"""
        if object_name:
            if object_name in self.reference_db:
                del self.reference_db[object_name]
                print(f"[FewShot] Cleared references for '{object_name}'")
        else:
            self.reference_db.clear()
            print("[FewShot] Cleared all references")
    
    def get_database_info(self):
        """Get info about stored references"""
        return {
            object_name: {
                'count': data['count'],
                'embedding_dim': data['embeddings'][0].shape[1] if data['embeddings'] else 0
            }
            for object_name, data in self.reference_db.items()
        }


class NavigationPipeline:
    def __init__(self, device='cpu'):
        # Use CPU - stable for all platforms
        self.device = 'cpu'
        print("[✓] Using CPU for all models (stable inference)")
        
        # Initialize FewShotMatcher for few-shot learning
        self.few_shot_matcher = None
        
        try:
            self.load_models()
            # Initialize FewShotMatcher after models are loaded
            self.few_shot_matcher = FewShotMatcher(device='cpu')
            print("[✓] FewShotMatcher initialized for few-shot learning")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[WARNING] Ignoring CUDA error during initialization: {e}")
            else:
                raise
    
    def get_device_str(self):
        """Get device as string - returns GPU or CPU based on availability"""
        return self.device
    
    def load_models(self):
        """Load all required models on CPU"""
        
        print("Loading GroundingDINO...")
        try:
            base_dir = Path(__file__).parent.parent
            config_path = str(base_dir / "GroundingDINO_SwinT_OGC.py")
            model_path = str(base_dir / "groundingdino_swint_ogc.pth")
            self.grounding_model = load_model(config_path, model_path)
            self.grounding_model.eval()
            
            # Keep on CPU
            for param in self.grounding_model.parameters():
                param.data = param.data.cpu()
            for buffer in self.grounding_model.buffers():
                buffer.data = buffer.data.cpu()
            
            print("[✓] GroundingDINO loaded on CPU")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] GroundingDINO CUDA error: {e}")
                self.grounding_model = None
            else:
                print(f"[✗] GroundingDINO failed: {e}")
                self.grounding_model = None
        
        print("Loading Depth Anything V2...")
        try:
            def load_depth():
                processor = AutoImageProcessor.from_pretrained(
                    "depth-anything/Depth-Anything-V2-base-hf"
                )
                model = AutoModelForDepthEstimation.from_pretrained(
                    "depth-anything/Depth-Anything-V2-base-hf"
                )
                return processor, model
            
            self.depth_processor, self.depth_model = load_with_timeout(
                load_depth, timeout_secs=60, model_name="Depth Anything V2"
            ) or (None, None)
            
            if self.depth_model is not None:
                self.depth_model.eval()
                try:
                    self.depth_model = self.depth_model.to('cpu')
                except:
                    pass
                print("[✓] Depth Anything V2 loaded on CPU")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Depth V2 CUDA error: {e}")
            else:
                print(f"[✗] Depth V2 failed: {e}")
            self.depth_model = None
            self.depth_processor = None
        print("Loading Whisper...")
        try:
            def load_whisper():
                return WhisperModel("small", device='cpu')
            
            self.whisper_model = load_with_timeout(
                load_whisper, timeout_secs=60, model_name="Whisper"
            )
            if self.whisper_model is not None:
                print("[✓] Whisper loaded")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Whisper CUDA error: {e}")
            else:
                print(f"[✗] Whisper failed: {e}")
            self.whisper_model = None
        
        print("Loading Flan-T5...")
        try:
            def load_t5():
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                return tokenizer, model
            
            result = load_with_timeout(load_t5, timeout_secs=60, model_name="Flan-T5")
            if result:
                self.instr_tokenizer, self.instr_model = result
                self.instr_model.eval()
                try:
                    self.instr_model = self.instr_model.to('cpu')
                except:
                    pass
                print("[✓] Flan-T5 loaded")
            else:
                self.instr_tokenizer = None
                self.instr_model = None
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] Flan-T5 CUDA error: {e}")
            else:
                print(f"[✗] Flan-T5 failed: {e}")
            self.instr_model = None
            self.instr_tokenizer = None
        
        print("Loading TTS...")
        try:
            def load_tts():
                # Lazy import of TTS to avoid slow scipy load on macOS
                global TTS
                if TTS is None:
                    print("  [TTS] Importing TTS module (this may take 30+ seconds)...")
                    from TTS.api import TTS as TTS_API
                    TTS = TTS_API
                return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            
            self.tts = load_with_timeout(load_tts, timeout_secs=120, model_name="TTS")
            if self.tts is not None:
                print("[SUCCESS] TTS loaded successfully")
                print("[✓] TTS loaded")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[✗] TTS CUDA error: {e}")
            else:
                print(f"[✗] TTS failed: {e}")
            self.tts = None
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Models Initialization Complete                    ║
║                                                              ║
║  GroundingDINO: {'✓ loaded' if self.grounding_model else '✗ failed'}                        
║  Depth Anything V2: {'✓ loaded' if self.depth_model else '✗ failed'}                   
║  Whisper: {'✓ loaded' if self.whisper_model else '✗ failed'}                          
║  Flan-T5: {'✓ loaded' if self.instr_model else '✗ failed'}                              
║  TTS: {'✓ loaded' if self.tts else '✗ failed'}                                
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        segments, info = self.whisper_model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        return text
    
    def extract_target_from_text(self, text):
        """Extract target object using simple, reliable heuristic (NO LLM)"""
        # Words to filter out (question words, articles, common verbs, prepositions)
        stop_words = {
            # Question/command words
            'find', 'show', 'where', 'what', 'when', 'why', 'how', 'can', 'could', 'would', 'will', 'do', 'does', 'did',
            # Articles
            'the', 'a', 'an', 'my', 'your', 'its', 'their', 'his', 'her', 'our',
            # Common verbs
            'is', 'are', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'locate', 'look', 'see', 'get', 'go', 'need',
            # Prepositions
            'to', 'at', 'in', 'on', 'for', 'from', 'by', 'with', 'of', 'about', 'up', 'down', 'out', 'over', 'under', 'between', 'through', 'during',
            # Pronouns/common words
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'please', 'or', 'and', 'but', 'not'
        }
        
        text_lower = text.lower()
        
        # Clean up punctuation
        text_lower = text_lower.replace('?', '').replace('!', '').replace('.', '').replace(',', '')
        
        # Split into words
        words = text_lower.split()
        
        # Filter out stop words - keep only meaningful nouns/objects
        target_words = [word.strip() for word in words if word.strip() and word.strip() not in stop_words]
        
        # Join remaining words into target
        target = ' '.join(target_words).strip()
        
        # If empty, return "object" as fallback
        if not target or len(target) == 0:
            target = "object"
        
        print(f"[TARGET EXTRACTION] Input: '{text}' → Output: '{target}'")
        return target
    
    def generate_instruction(self, target, steps, angle, distance_meters=None, confidence=None, depth=None, surfaces=None):
        """Generate natural language instruction without templates"""
        
        if distance_meters is None:
            distance_meters = steps * 0.75
        
        steps_int = int(round(steps))
        
        # Build context
        surface_info = ""
        if surfaces and len(surfaces) > 0:
            surface_names = ", ".join([s['surface'].lower() for s in surfaces])
            surface_info = f" It's on the {surface_names}."
        
        # Generate natural direction description from angle alone (no LLM if slow)
        # This provides 100% reliable, fast output without template text
        direction_text = self._describe_angle(angle)
        
        # Build natural instruction without any template format
        voice_instruction = f"The {target} is {steps_int} steps away.{surface_info} Turn {direction_text} and walk {steps_int} steps."
        
        # Detailed version for display
        detailed_instruction = (
            f"🎯 NAVIGATION FOR: {target.upper()}\n"
            f"{'='*50}\n"
            f"{voice_instruction}\n"
            f"{'='*50}"
        )
        
        return {
            'detailed': detailed_instruction,
            'conversational': voice_instruction,
            'summary': {
                'target': target,
                'distance_m': round(distance_meters, 2),
                'steps': steps_int,
                'direction': 'right' if angle > 0 else 'left' if angle < 0 else 'straight',
                'angle_degrees': round(angle, 1),
                'confidence_percent': round((confidence * 100) if confidence else 85, 1),
                'depth_m': round(depth if depth else 0, 3),
                'on_surface': surfaces[0]['surface'] if surfaces and len(surfaces) > 0 else None
            }
        }
    
    def _describe_angle(self, angle):
        """Convert angle to natural direction text - no numbers, pure natural language"""
        # Dead zone for "straight ahead"
        if abs(angle) <= 5:
            return "straight ahead"
        
        # Subtle turns (barely off center)
        if abs(angle) <= 12:
            if angle > 0:
                return "slightly to your right"  
            else:
                return "slightly to your left"
        
        # Moderate turns
        if abs(angle) <= 30:
            if angle > 0:
                return "to your right"
            else:
                return "to your left"
        
        # Sharp turns
        if angle > 0:
            return "sharply to your right"
        else:
            return "sharply to your left"
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        tts_path = "instruction.wav"
        self.tts.tts_to_file(text=text, file_path=tts_path)
        return tts_path
    
    def verify_models_on_cpu(self):
        """Verify that all models are on CPU by moving parameters directly (not using .to())"""
        try:
            # GroundingDINO
            if self.grounding_model is not None:
                for param in self.grounding_model.parameters():
                    if param.is_cuda or str(param.device) != 'cpu':
                        param.data = param.data.cpu()
                for buffer in self.grounding_model.buffers():
                    if buffer.is_cuda or str(buffer.device) != 'cpu':
                        buffer.data = buffer.data.cpu()
            
            # Depth model
            if self.depth_model is not None:
                try:
                    self.depth_model = self.depth_model.to('cpu')
                except:
                    pass
            
            # Instruction model
            if self.instr_model is not None:
                try:
                    self.instr_model = self.instr_model.to('cpu')
                except:
                    pass
                    
        except Exception as e:
            print(f"[WARNING] Could not verify models on CPU: {e}")
    
    def enhance_detection_caption(self, target):
        """Enhance detection caption for small gadgets and electronics"""
        target_lower = target.lower()
        
        # Mapping for small gadgets and electronics
        gadget_enhancements = {
            'speaker': 'speaker, audio speaker, Bluetooth speaker, wireless speaker, sound device',
            'headphones': 'headphones, earphones, earbuds, headset, audio headphones',
            'phone': 'phone, mobile phone, smartphone, cellular phone',
            'remote': 'remote, controller, remote control',
            'watch': 'watch, smartwatch, wristwatch, timepiece',
            'tablet': 'tablet, iPad, digital tablet',
            'charger': 'charger, power adapter, charging cable, USB charger',
            'cable': 'cable, cord, wire, charging cable',
            'plug': 'plug, power plug, electrical plug, adapter',
            'mouse': 'computer mouse, wireless mouse, mouse pad',
            'keyboard': 'keyboard, wireless keyboard, mechanical keyboard',
            'pen': 'pen, stylus, digital pen, writing pen',
            'lamp': 'lamp, desk lamp, table lamp, light',
            'glass': 'glass, drinking glass, water glass, cup',
            'bottle': 'bottle, water bottle, drinking bottle',
            'book': 'book, textbook, notebook',
            'keys': 'keys, key ring, set of keys',
            'wallet': 'wallet, purse, money holder',
        }
        
        # Check if target matches any gadget
        for gadget, description in gadget_enhancements.items():
            if gadget in target_lower:
                return description
        
        # Default: add "small" and "object with" prefix for better detection
        return f"{target}, small {target}, electronic {target}, device"
    
    def detect_spatial_relationships(self, image_np, target_bbox, target, image_tensor=None):
        """Detect if target object is on a surface using depth-aware spatial reasoning"""
        # Use image_tensor if available (more reliable) otherwise fall back to numpy
        use_tensor = image_tensor is not None
        
        # Optimized surface list - most common surfaces only (18 total for fast detection)
        # Ordered by likelihood - TABLE/DESK first (most common for objects)
        # FURNITURE FIRST (high-specificity), THEN GENERIC SURFACES
        surfaces = [
            # Most common for small objects like water bottles, phones, etc
            'table', 'desk', 'counter',
            # Other furniture
            'chair', 'shelf', 'cabinet', 'dresser', 'couch', 'sofa', 'bed', 'books',
            # Generic surfaces (low specificity - check after furniture)
            'floor', 'ground', 'grass', 'concrete',
            # Fallbacks
            'surface', 'level', 'pavement'
        ]
        
        detected_surfaces = []
        h, w = image_np.shape[:2]
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        target_height = target_y2 - target_y1
        target_width = target_x2 - target_x1
        target_area = target_height * target_width
        
        # Get depth map if available (for spatial awareness)
        try:
            depth_map = self.estimate_depth(image_np)
            if depth_map is not None and len(depth_map.shape) >= 2:
                # Get target's average depth
                depth_cropped = depth_map[max(0, int(target_y1)):min(h, int(target_y2)), 
                                             max(0, int(target_x1)):min(w, int(target_x2))]
                if depth_cropped.size > 0:
                    target_depth = np.mean(depth_cropped)
                else:
                    target_depth = None
            else:
                target_depth = None
        except:
            target_depth = None
        
        for surface in surfaces:
            try:
                # Standard thresholds - balance between accuracy and false positives
                detect_image = image_tensor if use_tensor else image_np
                surf_boxes, surf_logits, surf_phrases = self.predict_with_cpu_fallback(
                    model=self.grounding_model,
                    image=detect_image,
                    caption=surface,
                    box_threshold=0.35,   # Medium - catches real surfaces
                    text_threshold=0.30   # Medium - reasonable text match
                )
                
                if len(surf_boxes) > 0:
                    for i, surf_box in enumerate(surf_boxes):
                        surf_box = surf_box * torch.tensor([w, h, w, h])
                        surf_cx, surf_cy, surf_bw, surf_bh = surf_box
                        surf_x1 = int(surf_cx - surf_bw/2)
                        surf_y1 = int(surf_cy - surf_bh/2)
                        surf_x2 = int(surf_cx + surf_bw/2)
                        surf_y2 = int(surf_cy + surf_bh/2)
                        surf_area = surf_bw * surf_bh
                        
                        # Reasonable spatial reasoning
                        
                        # 1. Vertical: Object should be above or at surface level (not below)
                        above_or_on_surface = target_y2 <= surf_y2 + max(int(0.15*surf_bh), 25)
                        
                        # 2. Horizontal: Object should be roughly centered on surface
                        horiz_margin = max(0.35 * surf_bw, 45)  # Medium alignment tolerance
                        horizontal_alignment = (surf_x1 - horiz_margin <= target_cx <= surf_x2 + horiz_margin)
                        
                        # 3. Size: Object should be smaller than surface
                        target_smaller = target_area < 0.75 * surf_area
                        
                        # 4. Depth: If available, verify object is in front of or on surface
                        if target_depth is not None:
                            try:
                                depth_cropped = depth_map[max(0, int(surf_y1)):min(h, int(surf_y2)), 
                                                               max(0, int(surf_x1)):min(w, int(surf_x2))]
                                if depth_cropped.size > 0:
                                    surf_depth = np.mean(depth_cropped)
                                    # Object should be slightly closer than surface (small tolerance)
                                    depth_confirmed = target_depth <= surf_depth + 0.08 * surf_depth
                                else:
                                    depth_confirmed = True
                            except:
                                depth_confirmed = True
                        else:
                            depth_confirmed = True
                        
                        # ALL conditions must be true
                        is_on_surface = (above_or_on_surface and horizontal_alignment and target_smaller and depth_confirmed)
                        
                        if is_on_surface:
                            detection_confidence = float(surf_logits[i].item()) if surf_logits is not None and i < len(surf_logits) else 0.65
                            
                            # Boost confidence for high-specificity matches (books, pile, stack)
                            if surface in ['books', 'stack', 'pile']:
                                detection_confidence = min(0.95, detection_confidence + 0.15)
                            
                            detected_surfaces.append({
                                'surface': surface.capitalize(),
                                'confidence': detection_confidence
                            })
                            print(f"[SURFACE DETECTED ✓] '{surface}' under '{target}' (conf: {detection_confidence:.2f})")
                            break  # Only take first match per surface type (most specific wins)
            except Exception as e:
                print(f"[SURFACE] Error detecting '{surface}': {str(e)[:80]}")
                pass
            
            # EARLY STOPPING: If we already have 2 good surfaces, stop searching
            if len(detected_surfaces) >= 2:
                print(f"[SURFACE] Found {len(detected_surfaces)} surfaces, stopping early for speed")
                break
        
        # Sort by confidence and keep only top surface for cleaner output
        detected_surfaces.sort(key=lambda x: x['confidence'], reverse=True)
        top_surfaces = detected_surfaces[:1]  # Keep top 1 most confident surface (most relevant)
        
        if not top_surfaces:
            print(f"[SURFACE] No surfaces detected for '{target}'")
        else:
            print(f"[SURFACE] Returning top {len(top_surfaces)} surface(s) for '{target}'")
        
        return top_surfaces
    
    def improved_depth_to_steps(self, depth_meters, image_width, object_width_pixels):
        """
        Improved depth-to-steps conversion using calibration and object size
        
        Parameters:
        - depth_meters: estimated depth from depth model
        - image_width: width of the image in pixels
        - object_width_pixels: width of detected object in pixels
        
        Returns:
        - steps: estimated number of steps to reach the object
        """
        # Calibration parameters based on typical human height (1.7m)
        # and average step length (0.75m)
        
        # Improved conversion with multiple calibration factors
        # Account for depth estimation error at different distances
        
        if depth_meters < 0.5:
            distance = depth_meters * 0.8  # Closer objects have higher error
        elif depth_meters < 2:
            distance = depth_meters * 0.85
        elif depth_meters < 5:
            distance = depth_meters * 0.9
        else:
            distance = depth_meters * 0.95  # Distant objects more accurate
        
        # Apply object size heuristic for additional accuracy
        # Larger objects in frame likely mean they're closer
        object_ratio = object_width_pixels / image_width
        if object_ratio > 0.3:  # Large object
            distance *= 0.95
        elif object_ratio < 0.05:  # Very small object
            distance *= 1.05
        
        # Standard step length for adults (can be adjusted for user profile)
        # Average: 0.75m for normal walking
        average_step_length = 0.75
        
        steps = distance / average_step_length
        
        # Ensure minimum meaningful step count
        steps = max(steps, 1)
        
        return steps, distance
    
    def predict_with_cpu_fallback(self, model, image, caption, box_threshold=0.3, text_threshold=0.25):
        """
        Wrapper that ensures everything is on CPU with correct dtypes BEFORE calling predict.
        Enhanced with better small object detection support.
        """
        print(f"[PREDICT] Preparing for inference with caption: {caption}")
        
        # Ensure image is a proper tensor
        if isinstance(image, np.ndarray):
            print(f"[PREDICT] Converting numpy array (shape: {image.shape}) to tensor")
            # Normalize uint8 to [0, 1] if needed
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                print(f"[PREDICT] Normalized uint8 image to [0, 1] range")
            # Convert to tensor as-is (keep HWC format)
            image = torch.from_numpy(image).float()
            print(f"[PREDICT] Converted to tensor: shape={image.shape}, dtype={image.dtype}")
            image = image.to('cpu')
        elif hasattr(image, 'to'):
            image = image.to('cpu')
            print(f"[PREDICT] Image moved to CPU")
        
        # Verify format
        if hasattr(image, 'dtype'):
            if isinstance(image, torch.Tensor) and image.dtype not in [torch.float32, torch.float64]:
                print(f"[PREDICT] Converting image from {image.dtype} to float32")
                image = image.float()
        
        print(f"[PREDICT] Final image: dtype={image.dtype}, shape={image.shape}")
        
        # Ensure all model parameters are on CPU before predict is called
        try:
            for param in model.parameters():
                if param.is_cuda or str(param.device) != 'cpu':
                    param.data = param.data.cpu()
            for buffer in model.buffers():
                if buffer.is_cuda or str(buffer.device) != 'cpu':
                    buffer.data = buffer.data.cpu()
            print(f"[PREDICT] Model moved to CPU")
        except:
            print(f"[PREDICT] Could not move model to CPU, continuing anyway...")
        
        # Call predict
        print(f"[PREDICT] Calling predict() with device='cpu'")
        try:
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device='cpu'
            )
            print(f"[PREDICT] SUCCESS - found {len(boxes)} boxes")
        except Exception as e:
            print(f"[PREDICT] FAILED - {str(e)}")
            raise
        return boxes, logits, phrases
    
    def hybrid_detect_and_match(self, image_np, image_tensor, target, dino_boxes, dino_logits):
        """
        Hybrid detection: Use DINO boxes + Siamese few-shot matching for improved accuracy.
        
        If few-shot references exist, verify DINO detections using Siamese similarity.
        Returns enhanced confidence scores by combining both methods.
        """
        if not self.few_shot_matcher or len(self.few_shot_matcher.reference_db) == 0:
            # No few-shot references available, use DINO scores as-is
            return dino_logits
        
        h, w = image_np.shape[:2]
        enhanced_logits = dino_logits.clone() if isinstance(dino_logits, torch.Tensor) else dino_logits
        
        try:
            # For each DINO detection, try to match with few-shot learned objects
            for i, box in enumerate(dino_boxes):
                # Extract region around detected box
                box = box * torch.tensor([w, h, w, h])
                cx, cy, bw, bh = box
                x1 = max(0, int(cx - bw/2))
                y1 = max(0, int(cy - bh/2))
                x2 = min(w, int(cx + bw/2))
                y2 = min(h, int(cy + bh/2))
                
                # Extract this region
                region = image_np[y1:y2, x1:x2]
                
                if region.size == 0:
                    continue
                
                # Try few-shot matching
                matches = self.few_shot_matcher.match_in_region(region, similarity_threshold=0.6)
                
                if matches:
                    best_match = matches[0]
                    siamese_confidence = best_match['similarity']
                    
                    # Combine DINO confidence with Siamese confidence
                    dino_conf = float(dino_logits[i].item()) if isinstance(dino_logits[i], torch.Tensor) else float(dino_logits[i])
                    
                    # Average the confidences (Siamese match boosts confidence)
                    combined_conf = 0.6 * dino_conf + 0.4 * siamese_confidence
                    
                    if isinstance(enhanced_logits, torch.Tensor):
                        enhanced_logits[i] = torch.tensor(combined_conf)
                    else:
                        enhanced_logits[i] = combined_conf
                    
                    print(f"[HYBRID] DINO: {dino_conf:.3f}, Siamese ({best_match['object_name']}): {siamese_confidence:.3f} → Combined: {combined_conf:.3f}")
        
        except Exception as e:
            print(f"[HYBRID] Warning: Few-shot matching failed: {e}")
            # Fall back to DINO scores
        
        return enhanced_logits
    
    def process_image(self, image_path, target):
        """
        Process image and estimate navigation parameters
        Returns dict with success status and results
        Enhanced with spatial relationship detection and improved depth conversion
        """
        import time
        start_time = time.time()
        
        try:
            # Verify models are on CPU before starting
            self.verify_models_on_cpu()
            
            # Load image
            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            h, w, _ = img_np.shape
            
            # Ensure image tensor is on CPU and correct dtype
            if hasattr(image_tensor, 'to'):
                image_tensor = image_tensor.to('cpu')
            
            # Ensure image is float32 (critical for avoiding bool tensor errors)
            if isinstance(image_tensor, torch.Tensor):
                if image_tensor.dtype not in [torch.float32, torch.float64]:
                    print(f"[PROCESS] Converting image from {image_tensor.dtype} to float32")
                    image_tensor = image_tensor.float()
                print(f"[PROCESS] Image tensor: dtype={image_tensor.dtype}, shape={image_tensor.shape}")
            
            # Enhance detection caption for better small object detection
            enhanced_caption = self.enhance_detection_caption(target)
            print(f"[PROCESS] Enhanced caption: {enhanced_caption}")
            
            # Detect target using GroundingDINO with enhanced caption
            try:
                boxes, logits, phrases = self.predict_with_cpu_fallback(
                    model=self.grounding_model,
                    image=image_tensor,
                    caption=enhanced_caption,
                    box_threshold=0.25,  # Lowered threshold for small objects
                    text_threshold=0.2
                )
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] GroundingDINO prediction failed: {error_msg}")
                return {
                    'success': False,
                    'error': f'Object detection failed: {error_msg}'
                }
            
            if len(boxes) == 0:
                return {
                    'success': False,
                    'error': f'Target "{target}" not detected in image'
                }
            
            # Activate Siamese network: boost confidence using few-shot matching if available
            logits = self.hybrid_detect_and_match(img_np, image_tensor, target, boxes, logits)
            print(f"[PIPELINE] Post-Siamese confidence: {float(logits[0].item()) if torch.is_tensor(logits[0]) else float(logits[0])}")
            
            # Get bounding box
            box = boxes[0] * torch.tensor([w, h, w, h])
            cx, cy, bw, bh = box
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            
            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            object_width = x2 - x1
            
            # Estimate depth
            with torch.no_grad():
                try:
                    inputs = self.depth_processor(images=image_source, return_tensors="pt")
                    # Ensure inputs are on GPU if available
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = self.depth_model(**inputs)
                    depth = outputs.predicted_depth.to('cpu')
                except Exception as depth_error:
                    print(f"[ERROR] Depth estimation failed: {depth_error}")
                    return {
                        'success': False,
                        'error': f'Depth estimation error: {str(depth_error)}'
                    }
            
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            depth_map = depth.detach().cpu().numpy()
            obj_depth = depth_map[y1:y2, x1:x2].mean()
            
            # Use improved depth-to-steps conversion
            steps, meters = self.improved_depth_to_steps(obj_depth, w, object_width)
            
            # Calculate angle (raw, no thresholds)
            img_center = w / 2
            obj_center = (x1 + x2) / 2
            fov = 60  # Field of view in degrees
            angle = (obj_center - img_center) / w * fov
            
            # Detect spatial relationships (is object on a surface?)
            target_bbox = (x1, y1, x2, y2)
            surfaces = self.detect_spatial_relationships(img_np, target_bbox, target, image_tensor)
            
            # Draw visualization with enhanced styling
            vis = img_np.copy()
            
            # Draw semi-transparent overlay for better contrast
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 200, 255), -1)
            vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)
            
            # Draw thick bounding box with gradient effect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 200, 255), 4)
            # Inner glow effect
            cv2.rectangle(vis, (x1-2, y1-2), (x2+2, y2+2), (100, 220, 255), 1)
            
            # Draw centroid (target center)
            target_cx = (x1 + x2) // 2
            target_cy = (y1 + y2) // 2
            cv2.circle(vis, (target_cx, target_cy), 8, (50, 200, 255), -1)
            cv2.circle(vis, (target_cx, target_cy), 8, (255, 255, 0), 2)
            
            # Draw camera position indicator
            camera_cx = w // 2
            camera_cy = h // 2
            cv2.circle(vis, (camera_cx, camera_cy), 6, (0, 255, 100), -1)
            cv2.circle(vis, (camera_cx, camera_cy), 6, (255, 255, 255), 2)
            
            # Draw enhanced arrow from camera to target
            cv2.arrowedLine(vis, (camera_cx, camera_cy), (target_cx, target_cy), (50, 200, 255), 4, tipLength=0.25)
            # Arrow glow effect
            cv2.arrowedLine(vis, (camera_cx, camera_cy), (target_cx, target_cy), (150, 220, 255), 1, tipLength=0.25)
            
            # Create a CLEAN, spacious visualization with proper separation
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # ========== SECTION 1: TARGET (TOP) ==========
            section1_y = 0
            section1_h = 80
            cv2.rectangle(vis, (0, section1_y), (w, section1_h), (0, 255, 100), 4)  # Green border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section1_y), (w, section1_h), (20, 60, 20), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # TARGET label in left corner
            cv2.putText(vis, "TARGET:", (20, 35), font, 0.65, (150, 200, 150), 2)
            # TARGET value centered and large
            target_text = target.upper()
            target_w = cv2.getTextSize(target_text, font, 1.5, 3)[0][0]
            cv2.putText(vis, target_text, ((w - target_w) // 2, 65), font, 1.5, (100, 255, 100), 3)
            
            # ========== SECTION 2: DIRECTION (TOP-RIGHT) ==========
            section2_y = section1_h + 5
            section2_h = 80
            cv2.rectangle(vis, (0, section2_y), (w, section2_y + section2_h), (100, 150, 255), 4)  # Blue border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section2_y), (w, section2_y + section2_h), (20, 40, 60), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # DIRECTION label
            cv2.putText(vis, "DIRECTION:", (20, section2_y + 35), font, 0.65, (150, 180, 255), 2)
            
            # Determine direction
            if angle > 2:
                dir_text = "TURN RIGHT"
                dir_color = (100, 180, 255)
            elif angle < -2:
                dir_text = "TURN LEFT"
                dir_color = (0, 100, 255)
            else:
                dir_text = "GO STRAIGHT"
                dir_color = (100, 255, 100)
            
            dir_w = cv2.getTextSize(dir_text, font, 1.4, 3)[0][0]
            cv2.putText(vis, dir_text, ((w - dir_w) // 2, section2_y + 65), font, 1.4, dir_color, 3)
            
            # ========== SECTION 3: INFO (Distance & Steps) ==========
            section3_y = section2_y + section2_h + 5
            section3_h = 100
            cv2.rectangle(vis, (0, section3_y), (w, section3_y + section3_h), (150, 150, 100), 4)  # Gray border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section3_y), (w, section3_y + section3_h), (40, 40, 35), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # LEFT COLUMN: DISTANCE
            col_spacing = w // 2
            cv2.putText(vis, "DISTANCE:", (20, section3_y + 30), font, 0.65, (180, 180, 150), 2)
            distance_text = f"{meters:.1f}m"
            distance_w = cv2.getTextSize(distance_text, font, 1.3, 2)[0][0]
            cv2.putText(vis, distance_text, (20 + (col_spacing - 20 - distance_w) // 2, section3_y + 70), font, 1.3, (100, 255, 255), 2)
            
            # RIGHT COLUMN: STEPS
            cv2.putText(vis, "STEPS NEEDED:", (col_spacing + 20, section3_y + 30), font, 0.65, (180, 180, 150), 2)
            steps_text = f"{int(round(steps))}"
            steps_w = cv2.getTextSize(steps_text, font, 1.3, 2)[0][0]
            cv2.putText(vis, steps_text, (col_spacing + 20 + (col_spacing - 20 - steps_w) // 2, section3_y + 70), font, 1.3, (100, 255, 150), 2)
            
            # ========== SECTION 4: ACTION (BOTTOM) ==========
            section4_y = section3_y + section3_h + 5
            cv2.rectangle(vis, (0, section4_y), (w, h), (0, 255, 100), 4)  # Green border
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, section4_y), (w, h), (20, 60, 20), -1)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            
            # ACTION label
            cv2.putText(vis, "ACTION:", (20, section4_y + 35), font, 0.65, (150, 200, 150), 2)
            # ACTION instruction centered and large
            action_text = "WALK FORWARD"
            action_w = cv2.getTextSize(action_text, font, 1.6, 3)[0][0]
            cv2.putText(vis, action_text, ((w - action_w) // 2, section4_y + 75), font, 1.6, (100, 255, 100), 3)
            
            # Convert visualization to base64
            _, buffer = cv2.imencode('.png', vis)
            img_base64 = base64.b64encode(buffer).decode()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'target': target,
                'angle': float(angle),
                'steps': float(steps),
                'distance_meters': float(meters),
                'depth': float(obj_depth),
                'bbox': [x1, y1, x2, y2],
                'visualization': img_base64,
                'confidence': float(logits[0].item() if logits is not None else 0),
                'processing_time': float(processing_time),
                'surfaces': surfaces  # New: spatial relationship info
            }
        
        except Exception as e:
            print(f"[ERROR] process_image exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }
