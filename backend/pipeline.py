import os
import logging
import ctypes
import importlib.util

# Allow HuggingFace downloads by default, while letting callers override.
os.environ.setdefault('HF_HUB_OFFLINE', '0')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
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

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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
        logger.info("Added missing get_head_mask to BertModel")

    import inspect
    extended_mask_params = list(inspect.signature(BertModel.get_extended_attention_mask).parameters)
    if len(extended_mask_params) >= 4 and extended_mask_params[3] == "dtype":
        original_get_extended_attention_mask = BertModel.get_extended_attention_mask

        def get_extended_attention_mask_compat(self, attention_mask, input_shape, dtype=None):
            if isinstance(dtype, torch.device):
                dtype = None
            return original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)

        BertModel.get_extended_attention_mask = get_extended_attention_mask_compat
        logger.info("Patched BertModel.get_extended_attention_mask for GroundingDINO compatibility")
except Exception as e:
    logger.warning("Could not patch BertModel: %s", e)

# GroundingDINO
from groundingdino.util.inference import load_model, predict, load_image
import groundingdino.util.inference as groundingdino_inference

# FIX: Patch GroundingDINO's predict() to handle device properly
original_predict = groundingdino_inference.predict

def patched_predict(model, image, caption, box_threshold, text_threshold, device='cpu', remove_combined=False):
    """
    GroundingDINO predict variant that expects caller-managed model/image devices.
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
logger.info("GroundingDINO predict function patched")


# Audio LLM (faster-whisper)
from faster_whisper import WhisperModel

# Local instruction LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# TTS import is cached after first load.
TTS_API_CLASS = None


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
            logger.info("%s loaded successfully", model_name)
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            logger.warning("%s loading timed out after %ss", model_name, timeout_secs)
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

    def _prepare_image_tensor(self, image_input):
        """Convert an image-like input to a normalized BCHW tensor."""
        if isinstance(image_input, np.ndarray):
            image_tensor = torch.from_numpy(image_input).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
        elif isinstance(image_input, Image.Image):
            image_tensor = torch.from_numpy(np.array(image_input)).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
        elif isinstance(image_input, torch.Tensor):
            image_tensor = image_input.float()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)!r}")

        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def _compute_embedding(self, image_input):
        image_tensor = self._prepare_image_tensor(image_input)
        with torch.no_grad():
            return self.siamese(image_tensor.to(self.device))
    
    def add_reference(self, object_name, image_tensor):
        """
        Add reference image for few-shot learning.
        
        Args:
            object_name: Name of the object (e.g., "my_phone", "speaker")
            image_tensor: Input image as tensor (C, H, W) or PIL Image or numpy array
        
        Returns:
            embedding: The computed embedding for this reference
        """
        embedding = self._compute_embedding(image_tensor)
        
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
        
        try:
            query_embedding = self._compute_embedding(image_region)
        except TypeError:
            return []
        
        # Compare against all references
        matched_objects = []
        for object_name, data in self.reference_db.items():
            embeddings = torch.cat(data['embeddings'], dim=0).to(self.device)  # (num_refs, embedding_dim)
            
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

    def match_region_for_object(self, image_region, object_name):
        """Match an image region against one specific stored reference object."""
        if object_name not in self.reference_db:
            return None

        try:
            query_embedding = self._compute_embedding(image_region)
        except TypeError:
            return None

        embeddings = torch.cat(self.reference_db[object_name]['embeddings'], dim=0).to(self.device)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
        max_sim = similarities.max().item()
        return {
            'object_name': object_name,
            'similarity': float(max_sim),
            'num_references': self.reference_db[object_name]['count']
        }
    
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
    DEFAULT_DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE"
    DEFAULT_WHISPER_MODEL = "small"
    DEFAULT_INSTRUCTION_MODEL = "google/flan-t5-small"
    DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
    REFERENCE_SIMILARITY_WEIGHT = 0.6
    DINO_CONFIDENCE_WEIGHT = 0.4
    REFERENCE_RERANK_MAX_GAP = 0.08

    def __init__(self, device='auto'):
        self.requested_device = device or 'auto'
        self.torch_device = self._resolve_device(self.requested_device)
        self.device = str(self.torch_device)
        self.grounding_model = None
        self.depth_model = None
        self.depth_processor = None
        self.whisper_model = None
        self.whisper_device = None
        self.whisper_compute_type = None
        self.instr_tokenizer = None
        self.instr_model = None
        self.tts = None
        self.few_shot_matcher = None
        self.reference_catalog = {}
        self.model_status = {}
        self._dll_directory_handles = []

        self.depth_model_name = os.getenv("DEPTH_ANYTHING_MODEL", self.DEFAULT_DEPTH_MODEL)
        self.whisper_model_name = os.getenv("WHISPER_MODEL", self.DEFAULT_WHISPER_MODEL)
        self.instruction_model_name = os.getenv("INSTRUCTION_MODEL", self.DEFAULT_INSTRUCTION_MODEL)
        self.tts_model_name = os.getenv("TTS_MODEL", self.DEFAULT_TTS_MODEL)

        logger.info("NavigationPipeline using device=%s", self.device)
        self.load_models()
        self._load_few_shot_matcher()

    def _resolve_device(self, requested_device):
        if isinstance(requested_device, torch.device):
            requested = requested_device.type
        else:
            requested = str(requested_device or 'auto').lower()

        if requested in ('auto', 'gpu', 'cuda'):
            if torch.cuda.is_available():
                return torch.device('cuda')
            if requested in ('gpu', 'cuda'):
                logger.warning("CUDA was requested but is not available; falling back to CPU")
            return torch.device('cpu')

        if requested == 'cpu':
            return torch.device('cpu')

        logger.warning("Unknown device '%s'; falling back to auto selection", requested_device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_device_str(self):
        """Get the active torch device as a string."""
        return self.device
    
    def load_models(self):
        """Load the models used by the pipeline."""
        self.grounding_model = self._try_load_model(
            "GroundingDINO",
            self._load_grounding_model,
            timeout_secs=60,
        )
        self.depth_model = self._try_load_model(
            "Depth Anything 3",
            self._load_depth_model,
            timeout_secs=180,
        )
        self.whisper_model = self._try_load_model(
            "Whisper",
            self._load_whisper_model,
            timeout_secs=60,
        )

        instruction_result = self._try_load_model(
            "Flan-T5",
            self._load_instruction_model,
            timeout_secs=60,
        )
        if instruction_result is not None:
            self.instr_tokenizer, self.instr_model = instruction_result

        self.tts = self._try_load_model(
            "TTS",
            self._load_tts_model,
            timeout_secs=120,
        )
        self._log_model_summary()

    def _try_load_model(self, label, loader, timeout_secs):
        logger.info("Loading %s...", label)
        try:
            result = load_with_timeout(loader, timeout_secs=timeout_secs, model_name=label)
            if result is None:
                raise TimeoutError(f"{label} did not finish loading within {timeout_secs}s")
            self.model_status[label] = "loaded"
            logger.info("%s ready on %s", label, self._model_device_label(label))
            return result
        except Exception as exc:
            self.model_status[label] = f"failed: {exc}"
            logger.error("%s failed to load: %s", label, exc)
            return None

    def _model_device_label(self, label):
        if label == "Whisper" and self.whisper_device:
            return self.whisper_device
        return self.device

    def _load_grounding_model(self):
        base_dir = Path(__file__).parent.parent
        config_path = str(base_dir / "GroundingDINO_SwinT_OGC.py")
        model_path = str(base_dir / "groundingdino_swint_ogc.pth")
        model = load_model(config_path, model_path, device=self.device)
        model = model.to(self.torch_device)
        model.eval()
        return model

    def _load_depth_model(self):
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(self.depth_model_name)
        model = model.to(device=self.torch_device)
        model.eval()
        return model

    def _load_whisper_model(self):
        whisper_device = self._preferred_whisper_device()

        if whisper_device == "cuda" and os.name == "nt":
            self._add_windows_nvidia_dll_directories()
            if not self._windows_dll_available("cudnn_ops_infer64_8.dll"):
                logger.warning(
                    "Whisper CUDA runtime is missing cudnn_ops_infer64_8.dll; "
                    "using CPU for Whisper while keeping vision models on %s",
                    self.device,
                )
                whisper_device = "cpu"

        return self._create_whisper_model(whisper_device)

    def _preferred_whisper_device(self):
        requested = os.getenv("WHISPER_DEVICE", "auto").strip().lower()

        if requested in ("cuda", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return "cpu"

        return "cuda" if self.torch_device.type == "cuda" else "cpu"

    def _create_whisper_model(self, whisper_device):
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE")
        if not compute_type:
            compute_type = "float16" if whisper_device == "cuda" else "int8"

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            self.whisper_model_name,
            whisper_device,
            compute_type,
        )
        self.whisper_device = whisper_device
        self.whisper_compute_type = compute_type
        return WhisperModel(self.whisper_model_name, device=whisper_device, compute_type=compute_type)

    def _add_windows_nvidia_dll_directories(self):
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return

        for module_name in ("nvidia.cudnn", "nvidia.cublas"):
            try:
                spec = importlib.util.find_spec(module_name)
            except ModuleNotFoundError:
                spec = None

            if not spec or not spec.submodule_search_locations:
                continue

            for location in spec.submodule_search_locations:
                root = Path(location)
                for dll_dir in (root / "bin", root / "lib"):
                    if dll_dir.exists():
                        try:
                            self._dll_directory_handles.append(os.add_dll_directory(str(dll_dir)))
                            logger.info("Added DLL directory for %s: %s", module_name, dll_dir)
                        except OSError as exc:
                            logger.warning("Could not add DLL directory %s: %s", dll_dir, exc)

    def _windows_dll_available(self, dll_name):
        try:
            ctypes.WinDLL(dll_name)
            return True
        except OSError:
            return False

    def _load_instruction_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.instruction_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.instruction_model_name)
        model = model.to(self.torch_device)
        model.eval()
        return tokenizer, model

    def _load_tts_model(self):
        global TTS_API_CLASS
        if TTS_API_CLASS is None:
            from TTS.api import TTS as TTS_API
            TTS_API_CLASS = TTS_API
        return TTS_API_CLASS(
            model_name=self.tts_model_name,
            progress_bar=False,
            gpu=self.torch_device.type == "cuda",
        )

    def _load_few_shot_matcher(self):
        try:
            self.few_shot_matcher = FewShotMatcher(device=self.device)
            self.model_status["FewShotMatcher"] = "loaded"
            logger.info("FewShotMatcher ready on %s", self.device)
        except Exception as exc:
            self.few_shot_matcher = None
            self.model_status["FewShotMatcher"] = f"failed: {exc}"
            logger.error("FewShotMatcher failed to initialize: %s", exc)

    def _log_model_summary(self):
        summary = ", ".join(f"{name}={status}" for name, status in self.model_status.items())
        logger.info("Model load summary: %s", summary)

    def normalize_reference_name(self, name):
        normalized = " ".join(str(name or "").strip().lower().split())
        return normalized

    def has_reference(self, name):
        normalized = self.normalize_reference_name(name)
        return bool(
            self.few_shot_matcher
            and normalized
            and normalized in self.few_shot_matcher.reference_db
        )

    def register_reference_image(self, name, image_path, display_name=None):
        self._require_models(("few_shot_matcher", "FewShotMatcher"))
        normalized_name = self.normalize_reference_name(name)
        if not normalized_name:
            raise ValueError("Reference name cannot be empty")

        image = Image.open(image_path).convert("RGB")
        self.few_shot_matcher.clear_references(normalized_name)
        self.few_shot_matcher.add_reference(normalized_name, image)
        self.reference_catalog[normalized_name] = {
            'name': display_name or name,
            'normalized_name': normalized_name,
            'image_path': str(image_path),
        }
        return self.reference_catalog[normalized_name]

    def remove_reference_image(self, name):
        normalized_name = self.normalize_reference_name(name)
        if self.few_shot_matcher:
            self.few_shot_matcher.clear_references(normalized_name)
        self.reference_catalog.pop(normalized_name, None)

    def list_reference_images(self):
        references = []
        if not self.few_shot_matcher:
            return references

        db_info = self.few_shot_matcher.get_database_info()
        for normalized_name, info in db_info.items():
            catalog_entry = self.reference_catalog.get(normalized_name, {})
            references.append({
                'name': catalog_entry.get('name', normalized_name),
                'normalized_name': normalized_name,
                'image_path': catalog_entry.get('image_path'),
                'count': info.get('count', 0),
            })
        return sorted(references, key=lambda item: item['name'])

    def rerank_boxes_with_reference(self, image_np, target_name, dino_boxes, dino_logits):
        """Rerank candidate detections using the saved reference image for the target."""
        if (
            not self.few_shot_matcher
            or not target_name
            or not self.has_reference(target_name)
            or len(dino_boxes) == 0
        ):
            return dino_logits, []

        if torch.is_tensor(dino_logits):
            if len(dino_logits) < 2:
                logger.info(
                    "Skipping reference reranking for '%s': only %s DINO candidate available",
                    target_name,
                    len(dino_logits),
                )
                return dino_logits, []
            top_values = torch.topk(dino_logits, k=min(2, len(dino_logits))).values
            top_gap = float((top_values[0] - top_values[1]).item())
        else:
            if len(dino_logits) < 2:
                logger.info(
                    "Skipping reference reranking for '%s': only %s DINO candidate available",
                    target_name,
                    len(dino_logits),
                )
                return dino_logits, []
            sorted_logits = sorted((float(logit) for logit in dino_logits), reverse=True)
            top_gap = sorted_logits[0] - sorted_logits[1]

        if top_gap > self.REFERENCE_RERANK_MAX_GAP:
            logger.info(
                "Skipping reference reranking for '%s': top DINO gap %.4f exceeds threshold %.4f",
                target_name,
                top_gap,
                self.REFERENCE_RERANK_MAX_GAP,
            )
            return dino_logits, []

        normalized_name = self.normalize_reference_name(target_name)
        h, w = image_np.shape[:2]
        reranked_logits = dino_logits.clone() if torch.is_tensor(dino_logits) else list(dino_logits)
        reference_scores = []

        for idx, box in enumerate(dino_boxes):
            pixel_box = box * torch.tensor([w, h, w, h])
            cx, cy, bw, bh = pixel_box
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w, int(cx + bw / 2))
            y2 = min(h, int(cy + bh / 2))

            region = image_np[y1:y2, x1:x2]
            if region.size == 0:
                reference_scores.append({'candidate_index': idx, 'similarity': 0.0})
                continue

            match = self.few_shot_matcher.match_region_for_object(region, normalized_name)
            similarity = float(match['similarity']) if match else 0.0
            dino_conf = float(dino_logits[idx].item()) if torch.is_tensor(dino_logits) else float(dino_logits[idx])
            combined_score = (
                (self.DINO_CONFIDENCE_WEIGHT * dino_conf) +
                (self.REFERENCE_SIMILARITY_WEIGHT * similarity)
            )

            if torch.is_tensor(reranked_logits):
                reranked_logits[idx] = combined_score
            else:
                reranked_logits[idx] = combined_score

            reference_scores.append({
                'candidate_index': idx,
                'similarity': similarity,
                'dino_confidence': dino_conf,
                'combined_score': float(combined_score),
                'bbox': [x1, y1, x2, y2],
            })

        reference_scores.sort(key=lambda item: item['combined_score'], reverse=True)
        logger.info("Reference reranking for '%s': %s", normalized_name, reference_scores[:3])
        return reranked_logits, reference_scores

    def _require_models(self, *requirements):
        missing = [label for attr, label in requirements if getattr(self, attr, None) is None]
        if missing:
            raise RuntimeError("Required model(s) not loaded: " + ", ".join(missing))

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        self._require_models(("whisper_model", "Whisper"))
        try:
            return self._transcribe_with_whisper(audio_path)
        except Exception as exc:
            if not self._is_whisper_cuda_runtime_error(exc):
                raise

            logger.warning(
                "Whisper CUDA transcription failed: %s. Reloading Whisper on CPU and retrying.",
                exc,
            )
            self.whisper_model = self._create_whisper_model("cpu")
            self.model_status["Whisper"] = "loaded on cpu after CUDA fallback"
            return self._transcribe_with_whisper(audio_path)

    def _transcribe_with_whisper(self, audio_path):
        segments, info = self.whisper_model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        return text

    def _is_whisper_cuda_runtime_error(self, exc):
        if self.whisper_device != "cuda":
            return False

        message = str(exc).lower()
        return any(token in message for token in (
            "cuda",
            "cudnn",
            "cublas",
            "could not locate",
            "dll",
        ))
    
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
        self._require_models(("tts", "TTS"))
        tts_path = "instruction.wav"
        self.tts.tts_to_file(text=text, file_path=tts_path)
        return tts_path
    
    def estimate_depth(self, image):
        """Estimate a depth map with Depth Anything 3."""
        self._require_models(("depth_model", "Depth Anything 3"))
        prediction = self.depth_model.inference([image], export_format="mini_npz")
        depth_map = prediction.depth[0]
        return np.asarray(depth_map, dtype=np.float32)
    
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
                surf_boxes, surf_logits, surf_phrases = self.predict_with_model_device(
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
    
    def predict_with_model_device(self, model, image, caption, box_threshold=0.3, text_threshold=0.25):
        """Prepare tensors and run GroundingDINO on the configured device."""
        if model is None:
            raise RuntimeError("Required model(s) not loaded: GroundingDINO")

        logger.info("GroundingDINO inference caption=%s", caption)

        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).float()

        if isinstance(image, torch.Tensor):
            if image.dtype not in (torch.float32, torch.float64):
                image = image.float()
            image = image.to(self.torch_device)

        model = model.to(self.torch_device)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        logger.info("GroundingDINO found %s box(es)", len(boxes))
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
    
    def process_image(self, image_path, target, reference_name=None):
        """
        Process image and estimate navigation parameters
        Returns dict with success status and results
        Enhanced with spatial relationship detection and improved depth conversion
        """
        import time
        start_time = time.time()
        
        try:
            self._require_models(
                ("grounding_model", "GroundingDINO"),
                ("depth_model", "Depth Anything 3"),
            )
            
            # Load image
            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            h, w, _ = img_np.shape
            
            # Ensure image tensor is on the selected device and correct dtype
            if hasattr(image_tensor, 'to'):
                image_tensor = image_tensor.to(self.torch_device)
            
            if isinstance(image_tensor, torch.Tensor):
                if image_tensor.dtype not in [torch.float32, torch.float64]:
                    image_tensor = image_tensor.float()
            
            # Enhance detection caption for better small object detection
            enhanced_caption = self.enhance_detection_caption(target)
            logger.info("Enhanced detection caption=%s", enhanced_caption)
            
            # Detect target using GroundingDINO with enhanced caption
            try:
                boxes, logits, phrases = self.predict_with_model_device(
                    model=self.grounding_model,
                    image=image_tensor,
                    caption=enhanced_caption,
                    box_threshold=0.25,  # Lowered threshold for small objects
                    text_threshold=0.2
                )
            except Exception as e:
                error_msg = str(e)
                logger.error("GroundingDINO prediction failed: %s", error_msg)
                return {
                    'success': False,
                    'error': f'Object detection failed: {error_msg}'
                }
            
            if len(boxes) == 0:
                return {
                    'success': False,
                    'error': f'Target "{target}" not detected in image'
                }
            
            active_reference_name = reference_name if self.has_reference(reference_name) else None
            reference_scores = []
            if active_reference_name:
                logits, reference_scores = self.rerank_boxes_with_reference(
                    img_np,
                    active_reference_name,
                    boxes,
                    logits,
                )
            if logits is None or len(logits) == 0:
                return {
                    'success': False,
                    'error': f'Target "{target}" not detected reliably in image'
                }

            if torch.is_tensor(logits):
                best_idx = int(torch.argmax(logits).item())
                best_confidence = float(logits[best_idx].item())
            else:
                best_idx = max(range(len(logits)), key=lambda idx: float(logits[idx]))
                best_confidence = float(logits[best_idx])

            best_reference_score = None
            if reference_scores:
                best_reference_score = next(
                    (item for item in reference_scores if item['candidate_index'] == best_idx),
                    None
                )

            logger.info(
                "Selected detection index=%s confidence=%s reference=%s",
                best_idx,
                best_confidence,
                best_reference_score,
            )
            
            # Get the highest-confidence bounding box
            box = boxes[best_idx] * torch.tensor([w, h, w, h])
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
            
            try:
                depth_map = self.estimate_depth(img_np)
                if depth_map.shape[:2] != (h, w):
                    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
            except Exception as depth_error:
                logger.error("Depth Anything 3 estimation failed: %s", depth_error)
                return {
                    'success': False,
                    'error': f'Depth estimation error: {str(depth_error)}'
                }

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
                'confidence': float(best_confidence),
                'processing_time': float(processing_time),
                'surfaces': surfaces,  # New: spatial relationship info
                'reference_name': active_reference_name,
                'reference_score': best_reference_score,
            }
        
        except Exception as e:
            print(f"[ERROR] process_image exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }
