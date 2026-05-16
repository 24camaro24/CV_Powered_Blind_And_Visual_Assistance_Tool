import os
import logging
import ctypes
import importlib.util
import json

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
import time
import io
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


class ONNXFewShotLocalizer:
    """ONNX example-image object localizer used by the separate Find by Example flow."""

    def __init__(self, model_dir, device='cpu'):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install backend requirements before using Find by Example."
            ) from exc

        self.ort = ort
        self.model_dir = Path(model_dir)
        self.device = str(device)
        self.siamese_meta = self._load_meta("siamese_meta.json")
        self.localizer_meta = self._load_meta("localizer_meta.json")
        learned_threshold = float(
            self.siamese_meta.get("threshold", {}).get("learned_threshold", 0.5)
        )
        default_similarity_threshold = min(learned_threshold, 0.30)
        self.siamese_threshold = float(
            os.getenv("EXAMPLE_SIMILARITY_THRESHOLD", str(default_similarity_threshold))
        )
        self.abstain_threshold = float(
            self.localizer_meta.get("model_config", {}).get("abstain_threshold", 0.5)
        )
        self.min_localizer_score = float(os.getenv("EXAMPLE_MIN_LOCALIZER_SCORE", "0.0"))
        self.min_combined_confidence = float(os.getenv("EXAMPLE_MIN_COMBINED_CONFIDENCE", "0.0"))
        self.max_query_proposals = int(os.getenv("EXAMPLE_MAX_QUERY_PROPOSALS", "20"))
        self.max_localizer_candidates = int(os.getenv("EXAMPLE_MAX_LOCALIZER_CANDIDATES", "12"))
        self.segment_support_images = os.getenv("EXAMPLE_SEGMENT_SUPPORT", "0").strip().lower() in ("1", "true", "yes", "on")
        self.support_segment_min_fg = float(os.getenv("EXAMPLE_SUPPORT_MIN_FG_RATIO", "0.06"))
        self.rmbg_mask_threshold = float(os.getenv("EXAMPLE_RMBG_MASK_THRESHOLD", "0.5"))
        self.rmbg_size = int(os.getenv("EXAMPLE_RMBG_SIZE", "1024"))
        self.debug_save_proposals = os.getenv("EXAMPLE_DEBUG_SAVE_PROPOSALS", "1").strip().lower() in ("1", "true", "yes", "on")
        self.debug_root = Path(os.getenv("EXAMPLE_DEBUG_DIR", str(self.model_dir.parent / "backend" / "debug" / "example_proposals")))
        self.use_rembg_support = os.getenv("EXAMPLE_USE_REMBG", "0").strip().lower() in ("1", "true", "yes", "on")
        self.localizer_on_cpu = os.getenv("EXAMPLE_LOCALIZER_ON_CPU", "1").strip().lower() in ("1", "true", "yes", "on")
        self.rembg_remove = None
        logger.info(
            "ONNX few-shot config: similarity_threshold=%.4f segment_support=%s min_fg_ratio=%.4f",
            self.siamese_threshold,
            self.segment_support_images,
            self.support_segment_min_fg,
        )
        if self.use_rembg_support:
            try:
                from rembg import remove as rembg_remove
                self.rembg_remove = rembg_remove
                logger.info("Example support segmentation: rembg (U-2-Net) enabled")
            except Exception as exc:
                logger.warning("rembg unavailable for support segmentation: %s", exc)
        self.providers = self._select_providers()
        self.localizer_providers = ["CPUExecutionProvider"] if self.localizer_on_cpu else list(self.providers)
        self.rmbg_session = self._create_optional_rmbg_session()
        self.siamese_session = self._create_session(self.model_dir / "siamese.onnx")
        self.localizer_session = self._create_session(
            self.model_dir / "localizer.onnx",
            providers=self.localizer_providers,
        )
        logger.info(
            "ONNX example chain ready: RMBG=%s, Siamese=loaded, Localizer=loaded (%s)",
            "loaded" if self.rmbg_session is not None else "disabled (fallback preprocessing)",
            ",".join(self.localizer_providers),
        )

    def _load_meta(self, filename):
        path = self.model_dir / filename
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _select_providers(self):
        available = self.ort.get_available_providers()
        providers = []
        if self.device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers or available

    def _create_session(self, path, providers=None):
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {path}")

        session_options = self.ort.SessionOptions()
        session_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        selected_providers = list(providers) if providers is not None else list(self.providers)

        try:
            return self.ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=selected_providers,
            )
        except Exception as exc:
            if "CUDAExecutionProvider" in selected_providers:
                logger.warning(
                    "ONNX CUDA provider failed for %s: %s. Retrying on CPU.",
                    path.name,
                    exc,
                )
                selected_providers = ["CPUExecutionProvider"]
                if providers is None:
                    self.providers = ["CPUExecutionProvider"]
                return self.ort.InferenceSession(
                    str(path),
                    sess_options=session_options,
                    providers=selected_providers,
                )
            raise

    def _create_optional_rmbg_session(self):
        configured_path = os.getenv("EXAMPLE_RMBG_PATH", "").strip()
        if configured_path:
            rmbg_path = Path(configured_path)
        else:
            rmbg_path = self.model_dir / "rmbg.onnx"

        if not rmbg_path.exists():
            logger.info("ONNX RMBG model not found at %s; using fallback support preprocessing", rmbg_path)
            return None

        try:
            return self._create_session(rmbg_path)
        except Exception as exc:
            logger.warning("ONNX RMBG load failed (%s); using fallback support preprocessing", exc)
            return None

    def _is_cuda_oom_error(self, exc):
        text = str(exc).lower()
        return "cuda failure 2: out of memory" in text or "cuda out of memory" in text

    def _switch_example_sessions_to_cpu(self):
        if self.providers == ["CPUExecutionProvider"] and self.localizer_providers == ["CPUExecutionProvider"]:
            return

        logger.warning("ONNX CUDA OOM detected, switching example sessions to CPU and retrying once")
        self.providers = ["CPUExecutionProvider"]
        self.localizer_providers = ["CPUExecutionProvider"]
        self.rmbg_session = self._create_optional_rmbg_session()
        self.siamese_session = self._create_session(self.model_dir / "siamese.onnx")
        self.localizer_session = self._create_session(
            self.model_dir / "localizer.onnx",
            providers=self.localizer_providers,
        )

    def _run_session_with_oom_fallback(self, session_attr, inputs, stage_name):
        session = getattr(self, session_attr)
        try:
            return session.run(None, inputs)
        except Exception as exc:
            if self._is_cuda_oom_error(exc):
                logger.warning("ONNX %s stage hit CUDA OOM: %s", stage_name, exc)
                self._switch_example_sessions_to_cpu()
                session = getattr(self, session_attr)
                return session.run(None, inputs)
            raise

    def localize(self, support_images, query_image, debug_label="example"):
        if not support_images:
            raise ValueError("At least one support image is required")

        query_pil = self._to_pil(query_image)
        query_w, query_h = query_pil.size
        raw_support_images = [self._to_pil(s).copy() for s in support_images]
        if self.segment_support_images:
            support_images = self._prepare_support_images(raw_support_images)
        else:
            support_images = raw_support_images

        logger.info(
            "Example support preprocessing: enabled=%s supports=%d",
            self.segment_support_images,
            len(support_images),
        )

        debug_run_dir = self._create_debug_run_dir(debug_label) if self.debug_save_proposals else None
        if debug_run_dir is not None:
            self._save_support_debug_images(debug_run_dir, raw_support_images, prefix="support_raw")
            self._save_support_debug_images(debug_run_dir, support_images, prefix="support_preprocessed")
            query_path = debug_run_dir / "query_image.png"
            query_pil.save(query_path)
            logger.info("Example debug query image saved: %s", query_path)

        k_max = int(self.siamese_meta.get("k_max", 10))
        if len(support_images) > k_max:
            logger.info("Example support count exceeds k_max=%d; truncating to first %d images", k_max, k_max)
            support_images = support_images[:k_max]

        siamese_inputs, _ = self._build_inputs(support_images, query_pil, self.siamese_meta)
        existence_outputs = self._run_session_with_oom_fallback(
            "siamese_session",
            siamese_inputs,
            "siamese",
        )
        best_existence = float(np.asarray(existence_outputs[0]).reshape(-1)[0])
        logger.info(
            "Example siamese existence_prob=%.4f threshold=%.4f",
            best_existence,
            self.siamese_threshold,
        )
        if best_existence < self.siamese_threshold:
            return {
                "found": False,
                "reason": "similarity_below_threshold",
                "existence_prob": best_existence,
                "existence_threshold": self.siamese_threshold,
                "proposal_count": 1,
                "debug_dir": str(debug_run_dir) if debug_run_dir is not None else None,
            }

        localizer_inputs, query_transform = self._build_inputs(
            support_images,
            query_pil,
            self.localizer_meta,
        )
        raw_outputs = self._run_session_with_oom_fallback(
            "localizer_session",
            localizer_inputs,
            "localizer",
        )
        output_names = [output.name for output in self.localizer_session.get_outputs()]
        outputs = dict(zip(output_names, raw_outputs))

        best_box = np.asarray(outputs.get("best_box", raw_outputs[0])).reshape(-1)[:4]
        best_score = float(np.asarray(outputs.get("best_score", raw_outputs[1])).reshape(-1)[0])
        bg_default = np.array([0.0], dtype=np.float32)
        best_bg_prob = float(np.asarray(outputs.get("bg_prob", bg_default)).reshape(-1)[0])
        best_bbox = self._box_to_native_xyxy(best_box, query_transform)
        best_combined = float(min(best_existence, best_score))
        logger.info(
            "Example localizer output existence=%.4f localizer=%.4f bg=%.4f combined=%.4f bbox=%s",
            best_existence,
            best_score,
            best_bg_prob,
            best_combined,
            best_bbox,
        )

        if best_bg_prob >= self.abstain_threshold:
            return {
                "found": False,
                "reason": "localizer_abstained",
                "existence_prob": best_existence,
                "existence_threshold": self.siamese_threshold,
                "localizer_score": best_score,
                "bg_prob": best_bg_prob,
                "abstain_threshold": self.abstain_threshold,
                "combined_confidence": best_combined,
                "proposal_count": 1,
                "debug_dir": str(debug_run_dir) if debug_run_dir is not None else None,
            }

        if best_score < self.min_localizer_score:
            return {
                "found": False,
                "reason": "localizer_low_score",
                "existence_prob": best_existence,
                "existence_threshold": self.siamese_threshold,
                "localizer_score": best_score,
                "localizer_min_score": self.min_localizer_score,
                "combined_confidence": best_combined,
                "combined_min_confidence": self.min_combined_confidence,
                "bg_prob": best_bg_prob,
                "abstain_threshold": self.abstain_threshold,
                "proposal_count": 1,
                "debug_dir": str(debug_run_dir) if debug_run_dir is not None else None,
            }

        if best_combined < self.min_combined_confidence:
            return {
                "found": False,
                "reason": "combined_low_confidence",
                "existence_prob": best_existence,
                "existence_threshold": self.siamese_threshold,
                "localizer_score": best_score,
                "localizer_min_score": self.min_localizer_score,
                "combined_confidence": best_combined,
                "combined_min_confidence": self.min_combined_confidence,
                "bg_prob": best_bg_prob,
                "abstain_threshold": self.abstain_threshold,
                "proposal_count": 1,
                "debug_dir": str(debug_run_dir) if debug_run_dir is not None else None,
            }

        return {
            "found": True,
            "bbox": best_bbox,
            "existence_prob": best_existence,
            "existence_threshold": self.siamese_threshold,
            "localizer_score": best_score,
            "localizer_min_score": self.min_localizer_score,
            "combined_confidence": best_combined,
            "combined_min_confidence": self.min_combined_confidence,
            "bg_prob": best_bg_prob,
            "abstain_threshold": self.abstain_threshold,
            "proposal_count": 1,
            "chosen_proposal_index": 0,
            "chosen_proposal_rect": [0, 0, query_w, query_h],
            "providers": self.providers,
            "debug_dir": str(debug_run_dir) if debug_run_dir is not None else None,
        }

    def _generate_query_proposals(self, query_image):
        pil_image = self._to_pil(query_image)
        img_rgb = np.asarray(pil_image)
        h, w = img_rgb.shape[:2]
        min_area_ratio = 0.01
        max_area_ratio = 0.90
        min_area = int(min_area_ratio * h * w)
        max_area = int(max_area_ratio * h * w)

        proposals = [{
            "index": 0,
            "rect": [0, 0, w, h],
            "offset": (0, 0),
            "image": pil_image,
        }]

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_rects = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if area < min_area or area > max_area:
                continue
            if cw < 24 or ch < 24:
                continue
            contour_rects.append((x, y, cw, ch, area))

        contour_rects.sort(key=lambda item: item[4], reverse=True)
        next_index = 1
        for x, y, cw, ch, _ in contour_rects:
            if len(proposals) >= self.max_query_proposals:
                break

            margin_x = max(8, int(0.12 * cw))
            margin_y = max(8, int(0.12 * ch))
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + cw + margin_x)
            y2 = min(h, y + ch + margin_y)
            rect = [x1, y1, x2, y2]

            duplicate = any(self._rect_iou(rect, existing["rect"]) > 0.72 for existing in proposals)
            if duplicate:
                continue

            crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
            proposals.append({
                "index": next_index,
                "rect": rect,
                "offset": (x1, y1),
                "image": crop,
            })
            next_index += 1

        # Add deterministic grid proposals so salient misses do not hide valid objects.
        grid_rects = []
        for cols, rows in ((2, 2), (3, 2)):
            tile_w = max(1, w // cols)
            tile_h = max(1, h // rows)
            for row in range(rows):
                for col in range(cols):
                    x1 = col * tile_w
                    y1 = row * tile_h
                    x2 = w if col == cols - 1 else (col + 1) * tile_w
                    y2 = h if row == rows - 1 else (row + 1) * tile_h
                    grid_rects.append([x1, y1, x2, y2])

        # Center-focused proposals at different scales.
        for scale in (0.70, 0.50):
            cw = int(w * scale)
            ch = int(h * scale)
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
            x2 = min(w, x1 + cw)
            y2 = min(h, y1 + ch)
            grid_rects.append([x1, y1, x2, y2])

        for rect in grid_rects:
            if len(proposals) >= self.max_query_proposals:
                break
            duplicate = any(self._rect_iou(rect, existing["rect"]) > 0.72 for existing in proposals)
            if duplicate:
                continue
            x1, y1, x2, y2 = rect
            crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
            proposals.append({
                "index": next_index,
                "rect": rect,
                "offset": (x1, y1),
                "image": crop,
            })
            next_index += 1

        return proposals

    def _rect_iou(self, rect_a, rect_b):
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _create_debug_run_dir(self, debug_label):
        safe_label = "".join(ch for ch in str(debug_label) if ch.isalnum() or ch in ("_", "-")).strip()
        if not safe_label:
            safe_label = "example"
        run_id = f"{int(time.time() * 1000)}_{safe_label}"
        run_dir = self.debug_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Example debug artifacts directory: %s", run_dir)
        return run_dir

    def _prepare_support_images(self, support_images):
        processed = []
        for idx, support in enumerate(support_images):
            pil_img = self._to_pil(support)
            if self.rmbg_session is not None:
                processed_img, fg_ratio = self._segment_support_foreground_onnx(pil_img)
                if fg_ratio > 0.0:
                    logger.info(
                        "Support idx=%d RMBG crop foreground ratio=%.4f (accepted)",
                        idx,
                        fg_ratio,
                    )
                    processed.append(processed_img)
                else:
                    logger.info(
                        "Support idx=%d RMBG crop produced no foreground; fallback to raw image",
                        idx,
                    )
                    processed.append(pil_img)
                continue

            if self.rembg_remove is not None:
                segmented_img, fg_ratio = self._segment_support_foreground_rembg(pil_img)
            else:
                segmented_img, fg_ratio = self._segment_support_foreground(pil_img)

            if fg_ratio >= self.support_segment_min_fg:
                logger.info("Support idx=%d fallback segmentation ratio=%.4f (accepted)", idx, fg_ratio)
                processed.append(segmented_img)
            else:
                logger.info(
                    "Support idx=%d fallback segmentation ratio=%.4f below min=%.4f (raw kept)",
                    idx,
                    fg_ratio,
                    self.support_segment_min_fg,
                )
                processed.append(pil_img)
        return processed

    def _segment_support_foreground_rembg(self, pil_img):
        img_rgba = pil_img.convert("RGBA")
        try:
            removed = self.rembg_remove(img_rgba)
        except Exception as exc:
            logger.warning("rembg support segmentation failed: %s", exc)
            return pil_img, 0.0

        if isinstance(removed, bytes):
            try:
                removed = Image.open(io.BytesIO(removed))
            except Exception as exc:
                logger.warning("rembg returned unreadable bytes: %s", exc)
                return pil_img, 0.0

        removed = removed.convert("RGBA")
        arr = np.asarray(removed)
        alpha = arr[:, :, 3]
        fg_mask = np.where(alpha > 16, 255, 0).astype(np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        fg_ratio = float(np.count_nonzero(fg_mask)) / float(fg_mask.size)

        rgb = np.asarray(pil_img.convert("RGB")).copy()
        rgb[fg_mask == 0] = (114, 114, 114)
        return Image.fromarray(rgb).convert("RGB"), fg_ratio

    def _segment_support_foreground_onnx(self, pil_img):
        img_rgb = np.asarray(pil_img.convert("RGB"))
        h, w = img_rgb.shape[:2]
        if h < 8 or w < 8:
            return pil_img, 0.0

        resized = cv2.resize(img_rgb, (self.rmbg_size, self.rmbg_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        input_tensor = (input_tensor - mean) / std
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        try:
            input_name = self.rmbg_session.get_inputs()[0].name
            output = self.rmbg_session.run(None, {input_name: input_tensor})[0]
        except Exception as exc:
            logger.warning("ONNX RMBG inference failed: %s", exc)
            return pil_img, 0.0

        mask = np.asarray(output).reshape(self.rmbg_size, self.rmbg_size)
        foreground = mask > self.rmbg_mask_threshold
        if not np.any(foreground):
            return pil_img, 0.0

        ys, xs = np.where(foreground)
        mask_x1, mask_x2 = int(xs.min()), int(xs.max())
        mask_y1, mask_y2 = int(ys.min()), int(ys.max())

        x1 = int(np.floor(mask_x1 * w / self.rmbg_size))
        y1 = int(np.floor(mask_y1 * h / self.rmbg_size))
        x2 = int(np.ceil((mask_x2 + 1) * w / self.rmbg_size))
        y2 = int(np.ceil((mask_y2 + 1) * h / self.rmbg_size))

        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, x1 + 1, w))
        y2 = int(np.clip(y2, y1 + 1, h))

        fg_ratio = float(np.count_nonzero(foreground)) / float(foreground.size)
        cropped = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
        return cropped, fg_ratio

    def _segment_support_foreground(self, pil_img):
        img_rgb = np.asarray(pil_img.convert("RGB"))
        h, w = img_rgb.shape[:2]
        if h < 8 or w < 8:
            return pil_img, 0.0

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        margin_x = max(2, int(0.05 * w))
        margin_y = max(2, int(0.05 * h))
        rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))

        try:
            cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        except Exception as exc:
            logger.warning("Support segmentation grabCut failed: %s", exc)
            return pil_img, 0.0

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        fg_ratio = float(np.count_nonzero(fg_mask)) / float(fg_mask.size)

        segmented = img_rgb.copy()
        segmented[fg_mask == 0] = (114, 114, 114)
        segmented_pil = Image.fromarray(segmented).convert("RGB")
        return segmented_pil, fg_ratio

    def _save_support_debug_images(self, run_dir, support_images, prefix="support"):
        for idx, support in enumerate(support_images):
            try:
                img = self._to_pil(support)
                save_path = run_dir / f"{prefix}_{idx:02d}.png"
                img.save(save_path)
            except Exception as exc:
                logger.warning("Could not save support debug image idx=%d: %s", idx, exc)

    def _save_single_proposal_crop(self, run_dir, proposal, existence_prob):
        try:
            save_path = run_dir / (
                f"proposal_{proposal['index']:02d}_"
                f"exist_{int(round(existence_prob * 1000)):04d}.png"
            )
            proposal["image"].save(save_path)
        except Exception as exc:
            logger.warning("Could not save proposal crop idx=%d: %s", proposal.get("index", -1), exc)

    def _save_proposal_overview(self, run_dir, query_pil, proposals):
        try:
            vis = cv2.cvtColor(np.asarray(query_pil), cv2.COLOR_RGB2BGR)
            for proposal in proposals:
                x1, y1, x2, y2 = proposal["rect"]
                idx = proposal["index"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    vis,
                    str(idx),
                    (x1 + 4, max(14, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imwrite(str(run_dir / "query_proposals_overview.png"), vis)
        except Exception as exc:
            logger.warning("Could not save proposal overview: %s", exc)

    def _build_inputs(self, support_images, query_image, meta):
        img_size = int(meta.get("img_size", meta["inputs"]["query_img"]["shape"][-1]))
        k_max = int(meta.get("k_max", meta["inputs"]["support_imgs"]["shape"][1]))

        support_batch = np.zeros((1, k_max, 3, img_size, img_size), dtype=np.float32)
        support_mask = np.zeros((1, k_max), dtype=np.float32)

        for index, image in enumerate(support_images[:k_max]):
            support_batch[0, index] = self._letterbox(image, meta)[0]
            support_mask[0, index] = 1.0

        query_tensor, query_transform = self._letterbox(query_image, meta)
        query_tensor = query_tensor[np.newaxis, ...].astype(np.float32)

        return {
            "support_imgs": support_batch,
            "support_mask": support_mask,
            "query_img": query_tensor,
        }, query_transform

    def _letterbox(self, image, meta):
        img_size = int(meta.get("img_size", meta["inputs"]["query_img"]["shape"][-1]))
        pad_color = tuple(meta.get("preprocessing", {}).get("letterbox", {}).get(
            "pad_color_rgb",
            [114, 114, 114],
        ))
        pil_image = self._to_pil(image)
        orig_w, orig_h = pil_image.size
        scale = img_size / max(orig_w, orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_left = (img_size - new_w) // 2
        pad_top = (img_size - new_h) // 2

        resized = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (img_size, img_size), pad_color)
        canvas.paste(resized, (pad_left, pad_top))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = np.transpose(array, (2, 0, 1)).astype(np.float32)
        transform = {
            "img_size": img_size,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }
        return tensor, transform

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _box_to_native_xyxy(self, box, transform):
        img_size = transform["img_size"]
        scale = transform["scale"]
        pad_left = transform["pad_left"]
        pad_top = transform["pad_top"]
        orig_w = transform["orig_w"]
        orig_h = transform["orig_h"]

        cx, cy, width, height = [float(value) for value in box]
        cx_lb = cx * img_size
        cy_lb = cy * img_size
        width_lb = max(1.0, width * img_size)
        height_lb = max(1.0, height * img_size)

        x1 = (cx_lb - width_lb / 2 - pad_left) / scale
        y1 = (cy_lb - height_lb / 2 - pad_top) / scale
        x2 = (cx_lb + width_lb / 2 - pad_left) / scale
        y2 = (cy_lb + height_lb / 2 - pad_top) / scale

        x1 = int(np.clip(round(x1), 0, orig_w - 1))
        y1 = int(np.clip(round(y1), 0, orig_h - 1))
        x2 = int(np.clip(round(x2), x1 + 1, orig_w))
        y2 = int(np.clip(round(y2), y1 + 1, orig_h))
        return [x1, y1, x2, y2]


class NavigationPipeline:
    DEFAULT_DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE"
    DEFAULT_WHISPER_MODEL = "small"
    DEFAULT_INSTRUCTION_MODEL = "google/flan-t5-small"
    DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

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
        self.example_localizer = None
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
        self.example_localizer = self._try_load_model(
            "ONNX FewShotLocalizer",
            self._load_example_localizer,
            timeout_secs=120,
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
            logger.info("%s ready on %s", label, self._model_device_label(label, result))
            return result
        except Exception as exc:
            self.model_status[label] = f"failed: {exc}"
            logger.error("%s failed to load: %s", label, exc)
            return None

    def _model_device_label(self, label, loaded_result=None):
        if label == "Whisper" and self.whisper_device:
            return self.whisper_device
        if label == "ONNX FewShotLocalizer":
            source = loaded_result if loaded_result is not None else self.example_localizer
            providers = getattr(source, "providers", None) or []
            providers_text = ",".join(providers) if providers else "unknown"
            return f"onnx[{providers_text}]"
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

    def _load_example_localizer(self):
        model_dir = Path(__file__).parent.parent / "few_shot"
        localizer = ONNXFewShotLocalizer(model_dir=model_dir, device=self.device)
        available = localizer.ort.get_available_providers()
        selected = localizer.providers
        logger.info("ONNX providers available: %s", available)
        logger.info("ONNX providers selected: %s", selected)
        if self.torch_device.type == "cuda" and "CUDAExecutionProvider" not in selected:
            logger.warning(
                "Torch device is CUDA but ONNX is not using CUDAExecutionProvider; "
                "example flow will run on CPU."
            )
        return localizer

    def _get_example_localizer(self):
        if self.example_localizer is None:
            self.example_localizer = self._try_load_model(
                "ONNX FewShotLocalizer",
                self._load_example_localizer,
                timeout_secs=60,
            )
        if self.example_localizer is None:
            status = self.model_status.get("ONNX FewShotLocalizer", "not loaded")
            raise RuntimeError(f"ONNX FewShotLocalizer is unavailable: {status}")
        return self.example_localizer

    def _log_model_summary(self):
        summary = ", ".join(f"{name}={status}" for name, status in self.model_status.items())
        logger.info("Model load summary: %s", summary)

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

    def process_image_by_example(self, image_path, support_image_paths, target_label="example object"):
        """
        Locate an object in a scene using ONNX few-shot support images, then estimate navigation.
        This is intentionally separate from the text/GroundingDINO process_image flow.
        """
        import time
        start_time = time.time()

        try:
            self._require_models(("depth_model", "Depth Anything 3"))
            localizer = self._get_example_localizer()

            if not support_image_paths:
                return {
                    "success": False,
                    "error": "At least one example image is required",
                }

            scene_image = Image.open(image_path).convert("RGB")
            img_np = np.asarray(scene_image)
            support_images = [
                Image.open(path).convert("RGB")
                for path in support_image_paths
            ]

            localization = localizer.localize(support_images, scene_image, debug_label=target_label)
            if not localization.get("found"):
                reason = localization.get("reason", "not_found")
                reason_messages = {
                    "similarity_below_threshold": "Example object not found (support/query similarity too low)",
                    "localizer_abstained": "Example object not found (localizer abstained)",
                    "localizer_low_score": "Example match is too weak (localizer confidence too low)",
                    "combined_low_confidence": "Example match is too weak (combined confidence too low)",
                    "no_query_proposals": "No candidate regions were found in the query image",
                    "no_localizer_candidates": "No valid candidate regions passed to localizer",
                }
                return {
                    "success": False,
                    "error": reason_messages.get(reason, f"Example object not found ({reason})"),
                    "target": target_label,
                    "few_shot": localization,
                }

            bbox = localization["bbox"]
            result = self._navigation_from_example_bbox(
                img_np=img_np,
                bbox=bbox,
                target=target_label,
                localization=localization,
                processing_time=time.time() - start_time,
            )
            return result

        except Exception as exc:
            logger.error("Example-image processing failed: %s", exc)
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Example processing error: {str(exc)}",
            }

    def _navigation_from_example_bbox(self, img_np, bbox, target, localization, processing_time):
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        object_width = max(1, x2 - x1)

        try:
            depth_map = self.estimate_depth(img_np)
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as depth_error:
            logger.error("Depth Anything 3 estimation failed for example flow: %s", depth_error)
            return {
                "success": False,
                "error": f"Depth estimation error: {str(depth_error)}",
                "few_shot": localization,
            }

        depth_region = depth_map[y1:y2, x1:x2]
        if depth_region.size == 0:
            return {
                "success": False,
                "error": "Localized box is empty after clipping",
                "few_shot": localization,
            }

        obj_depth = float(np.nanmean(depth_region))
        if not np.isfinite(obj_depth):
            return {
                "success": False,
                "error": "Depth inside localized box is invalid",
                "few_shot": localization,
            }

        steps, meters = self.improved_depth_to_steps(obj_depth, w, object_width)

        img_center = w / 2
        obj_center = (x1 + x2) / 2
        fov = 60
        angle = (obj_center - img_center) / w * fov

        confidence = float(
            min(
                localization.get("existence_prob", 0.0),
                localization.get("localizer_score", 0.0),
            )
        )
        visualization = self._draw_example_visualization(
            img_np=img_np,
            bbox=(x1, y1, x2, y2),
            target=target,
            meters=meters,
            steps=steps,
            angle=angle,
            confidence=confidence,
        )

        return {
            "success": True,
            "source": "example",
            "target": target,
            "angle": float(angle),
            "steps": float(steps),
            "distance_meters": float(meters),
            "depth": obj_depth,
            "bbox": [x1, y1, x2, y2],
            "visualization": visualization,
            "confidence": confidence,
            "processing_time": float(processing_time),
            "surfaces": [],
            "few_shot": localization,
        }

    def _draw_example_visualization(self, img_np, bbox, target, meters, steps, angle, confidence):
        x1, y1, x2, y2 = bbox
        h, w = img_np.shape[:2]
        vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        box_color = (0, 190, 255)
        center_color = (60, 220, 80)
        text_color = (255, 255, 255)
        panel_color = (28, 70, 82)

        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        vis = cv2.addWeighted(vis, 0.82, overlay, 0.18, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 4)

        target_cx = (x1 + x2) // 2
        target_cy = (y1 + y2) // 2
        camera_cx = w // 2
        camera_cy = h // 2
        cv2.circle(vis, (target_cx, target_cy), 8, center_color, -1)
        cv2.circle(vis, (camera_cx, camera_cy), 6, (255, 255, 255), -1)
        cv2.arrowedLine(vis, (camera_cx, camera_cy), (target_cx, target_cy), box_color, 3, tipLength=0.25)

        panel_h = min(118, max(86, h // 5))
        panel = vis.copy()
        cv2.rectangle(panel, (0, 0), (w, panel_h), panel_color, -1)
        vis = cv2.addWeighted(vis, 0.68, panel, 0.32, 0)

        label = str(target or "example object").upper()[:28]
        direction = "STRAIGHT"
        if angle > 5:
            direction = "RIGHT"
        elif angle < -5:
            direction = "LEFT"

        cv2.putText(vis, f"EXAMPLE: {label}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(
            vis,
            f"{meters:.1f}m | {int(round(steps))} steps | {direction}",
            (18, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (160, 255, 220),
            2,
        )
        cv2.putText(
            vis,
            f"match {confidence * 100:.0f}%",
            (18, min(panel_h - 16, 104)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
        )

        _, buffer = cv2.imencode(".png", vis)
        return base64.b64encode(buffer).decode()
    
    def process_image(self, image_path, target):
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
            
            # Activate Siamese network: boost confidence using few-shot matching if available
            logits = self.hybrid_detect_and_match(img_np, image_tensor, target, boxes, logits)
            logger.info("Post-Siamese confidence=%s", float(logits[0].item()) if torch.is_tensor(logits[0]) else float(logits[0]))
            
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
