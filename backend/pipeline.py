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
import base64
import time
from pathlib import Path
import signal

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

# Local instruction LLM / CV processors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor

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


class NavigationPipeline:
    DEFAULT_DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE"
    DEFAULT_WHISPER_MODEL = "small"
    DEFAULT_INSTRUCTION_MODEL = "google/flan-t5-small"
    DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
    DEFAULT_OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"

    def __init__(self, device='auto'):
        self.requested_device = device or 'auto'
        self.torch_device = self._resolve_device(self.requested_device)
        self.device = str(self.torch_device)
        self.grounding_model = None
        self.owlv2_model = None
        self.owlv2_processor = None
        self.depth_model = None
        self.depth_processor = None
        self.whisper_model = None
        self.whisper_device = None
        self.whisper_compute_type = None
        self.instr_tokenizer = None
        self.instr_model = None
        self.tts = None
        self.few_shot_matcher = None
        self.model_status = {}
        self._dll_directory_handles = []

        self.depth_model_name = os.getenv("DEPTH_ANYTHING_MODEL", self.DEFAULT_DEPTH_MODEL)
        self.whisper_model_name = os.getenv("WHISPER_MODEL", self.DEFAULT_WHISPER_MODEL)
        self.instruction_model_name = os.getenv("INSTRUCTION_MODEL", self.DEFAULT_INSTRUCTION_MODEL)
        self.tts_model_name = os.getenv("TTS_MODEL", self.DEFAULT_TTS_MODEL)
        self.owlv2_model_name = os.getenv("OWLV2_MODEL", self.DEFAULT_OWLV2_MODEL)
        self.debug_enabled = os.getenv("NAV_DEBUG_SAVE", "1").strip().lower() in ("1", "true", "yes", "on")
        self.debug_root = Path(os.getenv("NAV_DEBUG_DIR", str(Path(__file__).parent / "debug")))
        if self.debug_enabled:
            try:
                self.debug_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Could not create debug directory %s: %s", self.debug_root, exc)
                self.debug_enabled = False

        logger.info("NavigationPipeline using device=%s", self.device)
        logger.info(
            "Debug artifacts: enabled=%s dir=%s",
            self.debug_enabled,
            self.debug_root,
        )
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
        owlv2_result = self._try_load_model(
            "OWLv2",
            self._load_owlv2_model,
            timeout_secs=180,
        )
        if owlv2_result is not None:
            self.owlv2_processor, self.owlv2_model = owlv2_result

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

    def _load_owlv2_model(self):
        model_cls = None
        try:
            from transformers import Owlv2ForObjectDetection
            model_cls = Owlv2ForObjectDetection
        except Exception:
            from transformers import AutoModelForZeroShotObjectDetection
            model_cls = AutoModelForZeroShotObjectDetection

        processor = AutoProcessor.from_pretrained(self.owlv2_model_name)
        model = model_cls.from_pretrained(self.owlv2_model_name)
        model = model.to(self.torch_device)
        model.eval()
        return processor, model

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

    def _normalize_object_name(self, object_name):
        normalized = " ".join(str(object_name or "").strip().lower().split())
        return normalized

    def load_support_references(self, object_name, support_images, replace_existing=True):
        """Load support/example images into the Siamese matcher for hybrid scoring."""
        if self.few_shot_matcher is None:
            raise RuntimeError("FewShotMatcher is unavailable")

        normalized_name = self._normalize_object_name(object_name)
        if not normalized_name:
            raise ValueError("Object name cannot be empty")

        if replace_existing:
            self.few_shot_matcher.clear_references(normalized_name)

        added_count = 0
        for support_image in support_images or []:
            if isinstance(support_image, (str, Path)):
                with Image.open(support_image) as pil_img:
                    image = pil_img.convert("RGB")
            else:
                image = support_image
            self.few_shot_matcher.add_reference(normalized_name, image)
            added_count += 1

        total_count = 0
        if normalized_name in self.few_shot_matcher.reference_db:
            total_count = int(self.few_shot_matcher.reference_db[normalized_name].get("count", 0))

        return {
            "object_name": normalized_name,
            "added_references": added_count,
            "total_references": total_count,
        }

    def _sanitize_debug_label(self, value, fallback="run"):
        text = str(value or "").strip().lower()
        safe = "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-"))
        return safe[:48] if safe else fallback

    def _create_debug_run_dir(self, mode, target=None):
        if not self.debug_enabled:
            return None

        mode_label = self._sanitize_debug_label(mode, fallback="mode")
        target_label = self._sanitize_debug_label(target, fallback="target")
        run_id = f"{int(time.time() * 1000)}_{mode_label}_{target_label}"
        run_dir = self.debug_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Debug artifacts directory: %s", run_dir)
        return run_dir

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def _debug_save_json(self, run_dir, filename, payload):
        if run_dir is None:
            return
        path = run_dir / filename
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._json_safe(payload), file, indent=2)

    def _debug_save_rgb_image(self, run_dir, filename, image):
        if run_dir is None:
            return
        pil_image = self._to_pil_image(image)
        pil_image.save(run_dir / filename)

    def _debug_save_bgr_image(self, run_dir, filename, image_bgr):
        if run_dir is None:
            return
        cv2.imwrite(str(run_dir / filename), image_bgr)

    def _debug_draw_candidates(self, scene_rgb, candidates, color=(0, 220, 255), header_text=None):
        vis = cv2.cvtColor(np.asarray(scene_rgb), cv2.COLOR_RGB2BGR)
        h, w = vis.shape[:2]
        for idx, candidate in enumerate(candidates or []):
            box = candidate.get("bbox", [0, 0, 1, 1])
            x1, y1, x2, y2 = self._clip_box_xyxy(box, w, h)
            score = float(candidate.get("score", 0.0))
            label = candidate.get("label", f"{idx}: {score:.3f}")
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                str(label),
                (x1 + 2, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

        if header_text:
            cv2.rectangle(vis, (0, 0), (w, 24), (20, 20, 20), -1)
            cv2.putText(
                vis,
                str(header_text)[:110],
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
        return vis

    def _to_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            with Image.open(image) as pil_img:
                return pil_img.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _clip_box_xyxy(self, box, width, height):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return [x1, y1, x2, y2]

    def _compute_iou_xyxy(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0

        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def _nms_candidates(self, candidates, iou_threshold=0.45, max_candidates=40):
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        kept = []
        for candidate in sorted_candidates:
            if len(kept) >= int(max_candidates):
                break
            keep = True
            for existing in kept:
                if self._compute_iou_xyxy(candidate["bbox"], existing["bbox"]) >= float(iou_threshold):
                    keep = False
                    break
            if keep:
                kept.append(candidate)
        return kept

    def _collect_owlv2_candidates(self, scene_image, support_images):
        self._require_models(("owlv2_model", "OWLv2"))
        if self.owlv2_processor is None:
            raise RuntimeError("OWLv2 processor is unavailable")
        if not support_images:
            return []

        scene_pil = self._to_pil_image(scene_image)
        support_pils = [self._to_pil_image(img) for img in support_images]

        owl_threshold = float(os.getenv("OWLV2_IMAGE_GUIDED_THRESHOLD", "0.12"))
        owl_nms = float(os.getenv("OWLV2_IMAGE_GUIDED_NMS", "0.30"))
        post_nms = float(os.getenv("OWLV2_POST_NMS", "0.45"))
        max_candidates = int(os.getenv("OWLV2_MAX_CANDIDATES", "40"))
        target_sizes = torch.tensor([(scene_pil.height, scene_pil.width)])

        candidates = []
        for support_idx, support_pil in enumerate(support_pils):
            try:
                inputs = self.owlv2_processor(
                    images=scene_pil,
                    query_images=support_pil,
                    return_tensors="pt",
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.torch_device)
                else:
                    inputs = {
                        key: (value.to(self.torch_device) if torch.is_tensor(value) else value)
                        for key, value in inputs.items()
                    }

                with torch.no_grad():
                    outputs = self.owlv2_model.image_guided_detection(**inputs)

                results = self.owlv2_processor.post_process_image_guided_detection(
                    outputs=outputs,
                    threshold=owl_threshold,
                    nms_threshold=owl_nms,
                    target_sizes=target_sizes,
                )
                if not results:
                    continue

                result = results[0]
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                for box, score in zip(boxes, scores):
                    score_value = float(score.item() if hasattr(score, "item") else score)
                    box_values = box.tolist() if hasattr(box, "tolist") else list(box)
                    candidates.append({
                        "bbox": [float(v) for v in box_values[:4]],
                        "score": score_value,
                        "support_index": support_idx,
                    })
            except Exception as exc:
                logger.warning("OWLv2 image-guided detection failed for support idx=%d: %s", support_idx, exc)

        return self._nms_candidates(candidates, iou_threshold=post_nms, max_candidates=max_candidates)

    def _dino_candidates_xyxy(self, boxes, logits, image_width, image_height):
        candidates = []
        if boxes is None or len(boxes) == 0:
            return candidates
        for idx, box in enumerate(boxes):
            scaled = box * torch.tensor([image_width, image_height, image_width, image_height])
            cx, cy, bw, bh = scaled
            x1 = float(cx - bw / 2)
            y1 = float(cy - bh / 2)
            x2 = float(cx + bw / 2)
            y2 = float(cy + bh / 2)
            score = float(logits[idx].item()) if torch.is_tensor(logits[idx]) else float(logits[idx])
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "score": score,
            })
        return candidates

    def _build_result_from_bbox(
        self,
        img_np,
        image_tensor,
        target,
        bbox_xyxy,
        confidence,
        start_time,
        source_mode="grounding_dino",
        debug_run_dir=None,
        debug_prefix="result",
    ):
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = self._clip_box_xyxy(bbox_xyxy, w, h)
        object_width = max(1, x2 - x1)

        try:
            depth_map = self.estimate_depth(img_np)
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as depth_error:
            logger.error("Depth Anything 3 estimation failed: %s", depth_error)
            return {
                "success": False,
                "error": f"Depth estimation error: {str(depth_error)}",
            }

        depth_region = depth_map[y1:y2, x1:x2]
        if depth_region.size == 0:
            return {
                "success": False,
                "error": "Detected target region is empty",
            }

        obj_depth = float(np.nanmean(depth_region))
        if not np.isfinite(obj_depth):
            return {
                "success": False,
                "error": "Detected depth is invalid",
            }

        steps, meters = self.improved_depth_to_steps(obj_depth, w, object_width)
        img_center = w / 2
        obj_center = (x1 + x2) / 2
        fov = 60
        angle = (obj_center - img_center) / w * fov

        surfaces = self.detect_spatial_relationships(
            img_np,
            (x1, y1, x2, y2),
            target,
            image_tensor,
        )

        vis = img_np.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 200, 255), 3)
        cv2.putText(
            vis,
            f"{target} ({confidence * 100:.0f}%)",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        _, buffer = cv2.imencode(".png", vis)
        img_base64 = base64.b64encode(buffer).decode()

        if debug_run_dir is not None:
            self._debug_save_rgb_image(debug_run_dir, f"{debug_prefix}_scene_rgb.png", img_np)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            self._debug_save_bgr_image(debug_run_dir, f"{debug_prefix}_bbox.png", vis_bgr)

        result = {
            "success": True,
            "source": source_mode,
            "target": target,
            "angle": float(angle),
            "steps": float(steps),
            "distance_meters": float(meters),
            "depth": float(obj_depth),
            "bbox": [x1, y1, x2, y2],
            "visualization": img_base64,
            "confidence": float(confidence),
            "processing_time": float(max(0.0, time.time() - start_time)),
            "surfaces": surfaces,
        }
        if debug_run_dir is not None:
            result["debug_dir"] = str(debug_run_dir)
        return result

    def process_image_with_owl_examples(self, image_path, support_image_paths, target_label=None):
        import time
        start_time = time.time()
        target_text = (target_label or "").strip() or "example object"
        run_dir = self._create_debug_run_dir("owlv2_examples", target_text)
        try:
            self._require_models(
                ("owlv2_model", "OWLv2"),
                ("depth_model", "Depth Anything 3"),
            )

            scene_pil = self._to_pil_image(image_path)
            support_images = [self._to_pil_image(path) for path in support_image_paths or []]
            self._debug_save_rgb_image(run_dir, "scene_image.png", scene_pil)
            for idx, support in enumerate(support_images):
                self._debug_save_rgb_image(run_dir, f"support_{idx:02d}.png", support)

            if len(support_images) == 0:
                self._debug_save_json(run_dir, "result.json", {
                    "success": False,
                    "error": "At least one example image is required for OWLv2 mode",
                    "mode": "owlv2_examples",
                })
                return {
                    "success": False,
                    "error": "At least one example image is required for OWLv2 mode",
                }

            candidates = self._collect_owlv2_candidates(scene_pil, support_images)
            debug_candidates = [
                {
                    **candidate,
                    "label": f"{idx}: {float(candidate.get('score', 0.0)):.3f} s{int(candidate.get('support_index', -1))}",
                }
                for idx, candidate in enumerate(candidates)
            ]
            candidates_vis = self._debug_draw_candidates(
                scene_pil,
                debug_candidates,
                color=(0, 220, 255),
                header_text=f"OWLv2 candidates: {len(candidates)}",
            )
            self._debug_save_bgr_image(run_dir, "owlv2_candidates.png", candidates_vis)

            if len(candidates) == 0:
                self._debug_save_json(run_dir, "result.json", {
                    "success": False,
                    "error": "No OWLv2 match found from example images",
                    "num_candidates": 0,
                    "mode": "owlv2_examples",
                })
                return {
                    "success": False,
                    "error": "No OWLv2 match found from example images",
                }

            best = max(candidates, key=lambda item: float(item.get("score", 0.0)))
            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            if hasattr(image_tensor, "to"):
                image_tensor = image_tensor.to(self.torch_device)
            if isinstance(image_tensor, torch.Tensor) and image_tensor.dtype not in [torch.float32, torch.float64]:
                image_tensor = image_tensor.float()

            result = self._build_result_from_bbox(
                img_np=img_np,
                image_tensor=image_tensor,
                target=target_text,
                bbox_xyxy=best["bbox"],
                confidence=float(best["score"]),
                start_time=start_time,
                source_mode="owlv2_image_guided",
                debug_run_dir=run_dir,
                debug_prefix="owlv2_best",
            )
            if result.get("success"):
                result["owlv2"] = {
                    "num_candidates": len(candidates),
                    "best_support_index": int(best.get("support_index", -1)),
                    "best_score": float(best.get("score", 0.0)),
                }
                self._debug_save_json(run_dir, "result.json", {
                    "mode": "owlv2_examples",
                    "target": target_text,
                    "best": best,
                    "num_candidates": len(candidates),
                    "result": result,
                })
            return result

        except Exception as exc:
            logger.error("OWLv2 example processing failed: %s", exc)
            self._debug_save_json(run_dir, "result.json", {
                "success": False,
                "mode": "owlv2_examples",
                "error": str(exc),
            })
            return {
                "success": False,
                "error": f"OWLv2 processing error: {str(exc)}",
            }

    def process_image_hybrid_owl_dino(self, image_path, support_image_paths, target):
        import time
        start_time = time.time()
        run_dir = self._create_debug_run_dir("hybrid_owl_dino", target)
        self._debug_save_rgb_image(run_dir, "scene_image.png", image_path)
        for idx, support_path in enumerate(support_image_paths or []):
            self._debug_save_rgb_image(run_dir, f"support_{idx:02d}.png", support_path)

        dino_result = self.process_image(image_path, target)
        owl_result = self.process_image_with_owl_examples(image_path, support_image_paths, target_label=target)

        self._debug_save_json(run_dir, "sub_results.json", {
            "dino_result": dino_result,
            "owl_result": owl_result,
        })

        dino_ok = bool(dino_result and dino_result.get("success"))
        owl_ok = bool(owl_result and owl_result.get("success"))

        if not dino_ok and not owl_ok:
            dino_error = dino_result.get("error") if isinstance(dino_result, dict) else "unknown"
            owl_error = owl_result.get("error") if isinstance(owl_result, dict) else "unknown"
            return {
                "success": False,
                "error": f"Hybrid detection failed (DINO: {dino_error}; OWLv2: {owl_error})",
                "debug_dir": str(run_dir) if run_dir is not None else None,
            }

        if dino_ok and not owl_ok:
            result = dict(dino_result)
            result["source"] = "hybrid_owl_dino_fallback_dino"
            result["hybrid"] = {
                "mode": "fallback_dino",
                "dino_confidence": float(dino_result.get("confidence", 0.0)),
                "owl_confidence": 0.0,
            }
            if run_dir is not None:
                result["debug_dir"] = str(run_dir)
                self._debug_save_json(run_dir, "result.json", result)
            return result

        if owl_ok and not dino_ok:
            result = dict(owl_result)
            result["source"] = "hybrid_owl_dino_fallback_owlv2"
            result["hybrid"] = {
                "mode": "fallback_owlv2",
                "dino_confidence": 0.0,
                "owl_confidence": float(owl_result.get("confidence", 0.0)),
            }
            if run_dir is not None:
                result["debug_dir"] = str(run_dir)
                self._debug_save_json(run_dir, "result.json", result)
            return result

        dino_conf = float(dino_result.get("confidence", 0.0))
        owl_conf = float(owl_result.get("confidence", 0.0))
        dino_weight = float(os.getenv("HYBRID_DINO_WEIGHT", "0.55"))
        owl_weight = float(os.getenv("HYBRID_OWL_WEIGHT", "0.45"))
        iou_threshold = float(os.getenv("HYBRID_OWL_DINO_IOU_THRESHOLD", "0.35"))

        dino_bbox = dino_result.get("bbox", [0, 0, 1, 1])
        owl_bbox = owl_result.get("bbox", [0, 0, 1, 1])
        iou = self._compute_iou_xyxy(dino_bbox, owl_bbox)

        if iou >= iou_threshold:
            score_sum = max(1e-6, dino_conf + owl_conf)
            alpha = dino_conf / score_sum
            beta = owl_conf / score_sum
            merged_bbox = [
                alpha * float(dino_bbox[0]) + beta * float(owl_bbox[0]),
                alpha * float(dino_bbox[1]) + beta * float(owl_bbox[1]),
                alpha * float(dino_bbox[2]) + beta * float(owl_bbox[2]),
                alpha * float(dino_bbox[3]) + beta * float(owl_bbox[3]),
            ]
            merged_conf = dino_weight * dino_conf + owl_weight * owl_conf

            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            if hasattr(image_tensor, "to"):
                image_tensor = image_tensor.to(self.torch_device)
            if isinstance(image_tensor, torch.Tensor) and image_tensor.dtype not in [torch.float32, torch.float64]:
                image_tensor = image_tensor.float()

            result = self._build_result_from_bbox(
                img_np=img_np,
                image_tensor=image_tensor,
                target=target,
                bbox_xyxy=merged_bbox,
                confidence=merged_conf,
                start_time=start_time,
                source_mode="hybrid_owl_dino",
                debug_run_dir=run_dir,
                debug_prefix="hybrid_merged",
            )
            if result.get("success"):
                result["hybrid"] = {
                    "mode": "merged_bbox",
                    "iou": float(iou),
                    "dino_confidence": dino_conf,
                    "owl_confidence": owl_conf,
                }
                self._debug_save_json(run_dir, "result.json", result)
            return result

        dino_weighted = dino_weight * dino_conf
        owl_weighted = owl_weight * owl_conf
        chosen = dict(dino_result if dino_weighted >= owl_weighted else owl_result)
        chosen["source"] = "hybrid_owl_dino"
        chosen["hybrid"] = {
            "mode": "winner_takes_all",
            "iou": float(iou),
            "dino_confidence": dino_conf,
            "owl_confidence": owl_conf,
            "winner": "dino" if dino_weighted >= owl_weighted else "owlv2",
        }
        if run_dir is not None:
            chosen["debug_dir"] = str(run_dir)
            self._debug_save_json(run_dir, "result.json", chosen)

            try:
                scene_pil = self._to_pil_image(image_path)
                vis = cv2.cvtColor(np.asarray(scene_pil), cv2.COLOR_RGB2BGR)
                h, w = vis.shape[:2]
                d_x1, d_y1, d_x2, d_y2 = self._clip_box_xyxy(dino_bbox, w, h)
                o_x1, o_y1, o_x2, o_y2 = self._clip_box_xyxy(owl_bbox, w, h)
                c_x1, c_y1, c_x2, c_y2 = self._clip_box_xyxy(chosen.get("bbox", [0, 0, 1, 1]), w, h)
                cv2.rectangle(vis, (d_x1, d_y1), (d_x2, d_y2), (255, 80, 80), 2)
                cv2.rectangle(vis, (o_x1, o_y1), (o_x2, o_y2), (80, 180, 255), 2)
                cv2.rectangle(vis, (c_x1, c_y1), (c_x2, c_y2), (80, 255, 120), 3)
                cv2.putText(vis, f"DINO {dino_conf:.3f}", (d_x1, max(14, d_y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                cv2.putText(vis, f"OWL {owl_conf:.3f}", (o_x1, max(30, o_y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                cv2.putText(vis, f"WINNER {chosen['hybrid']['winner']} (IoU {iou:.3f})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                self._debug_save_bgr_image(run_dir, "hybrid_boxes.png", vis)
            except Exception as exc:
                logger.warning("Could not save hybrid debug overlay: %s", exc)
        return chosen

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
        
        # Target-specific caption expansions.
        target_enhancements = {
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
            'passport': 'passport, passport booklet, travel document, booklet',
            'keys': 'keys, key ring, set of keys',
            'wallet': 'wallet, purse, money holder',
        }
        
        # Check if target matches any known object class
        for object_key, description in target_enhancements.items():
            if object_key in target_lower:
                return description
        
        # Generic fallback: keep caption neutral (avoid biasing to electronics).
        return f"{target}, {target} object, {target} item"
    
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
        target_key = self._normalize_object_name(target)
        target_exists = target_key in self.few_shot_matcher.reference_db if target_key else False
        similarity_threshold = float(os.getenv("HYBRID_SIMILARITY_THRESHOLD", "0.6"))
        target_match_threshold = float(os.getenv("HYBRID_TARGET_MATCH_THRESHOLD", "0.72"))
        target_mismatch_penalty = float(os.getenv("HYBRID_TARGET_MISMATCH_PENALTY", "0.15"))
        target_dino_weight = float(os.getenv("HYBRID_TARGET_DINO_WEIGHT", "0.35"))
        target_siamese_weight = float(os.getenv("HYBRID_TARGET_SIAMESE_WEIGHT", "0.65"))

        try:
            # For each DINO detection, try to match with few-shot learned objects
            for i, box in enumerate(dino_boxes):
                # Extract region around detected box
                box = box * torch.tensor([w, h, w, h])
                cx, cy, bw, bh = box
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(w, int(cx + bw / 2))
                y2 = min(h, int(cy + bh / 2))

                # Extract this region
                region = image_np[y1:y2, x1:x2]

                if region.size == 0:
                    continue

                # Try few-shot matching
                matches = self.few_shot_matcher.match_in_region(
                    region,
                    similarity_threshold=similarity_threshold,
                )
                if target_exists:
                    matches = [match for match in matches if match.get("object_name") == target_key]

                dino_conf = float(dino_logits[i].item()) if isinstance(dino_logits[i], torch.Tensor) else float(dino_logits[i])

                if matches:
                    best_match = matches[0]
                    siamese_confidence = best_match["similarity"]

                    if target_exists:
                        if siamese_confidence >= target_match_threshold:
                            combined_conf = target_dino_weight * dino_conf + target_siamese_weight * siamese_confidence
                        else:
                            combined_conf = target_mismatch_penalty * dino_conf
                    else:
                        combined_conf = 0.6 * dino_conf + 0.4 * siamese_confidence

                    if isinstance(enhanced_logits, torch.Tensor):
                        enhanced_logits[i] = torch.tensor(combined_conf)
                    else:
                        enhanced_logits[i] = combined_conf

                    print(f"[HYBRID] DINO: {dino_conf:.3f}, Siamese ({best_match['object_name']}): {siamese_confidence:.3f} -> Combined: {combined_conf:.3f}")
                elif target_exists:
                    # Target references exist but this box does not match target examples.
                    penalized_conf = target_mismatch_penalty * dino_conf
                    if isinstance(enhanced_logits, torch.Tensor):
                        enhanced_logits[i] = torch.tensor(penalized_conf)
                    else:
                        enhanced_logits[i] = penalized_conf

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
        run_dir = self._create_debug_run_dir("grounding_dino", target)
        
        try:
            self._require_models(
                ("grounding_model", "GroundingDINO"),
                ("depth_model", "Depth Anything 3"),
            )
            
            # Load image
            image_source, image_tensor = load_image(image_path)
            img_np = np.array(image_source)
            h, w, _ = img_np.shape
            self._debug_save_rgb_image(run_dir, "scene_image.png", img_np)
            
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
                error_payload = {
                    "success": False,
                    "mode": "grounding_dino",
                    "target": target,
                    "enhanced_caption": enhanced_caption,
                    "error": f"Object detection failed: {error_msg}",
                }
                self._debug_save_json(run_dir, "result.json", error_payload)
                return {
                    'success': False,
                    'error': f'Object detection failed: {error_msg}',
                    'debug_dir': str(run_dir) if run_dir is not None else None,
                }

            raw_candidates = self._dino_candidates_xyxy(boxes, logits, w, h)
            raw_debug_candidates = []
            for idx, candidate in enumerate(raw_candidates):
                phrase_text = ""
                if idx < len(phrases):
                    phrase_text = str(phrases[idx] or "").strip()
                raw_debug_candidates.append(
                    {
                        **candidate,
                        "label": f"{idx}: {float(candidate.get('score', 0.0)):.3f} {phrase_text[:24]}".strip(),
                    }
                )
            raw_candidates_vis = self._debug_draw_candidates(
                img_np,
                raw_debug_candidates,
                color=(30, 200, 255),
                header_text=f"DINO raw candidates: {len(raw_candidates)}",
            )
            self._debug_save_bgr_image(run_dir, "dino_candidates_raw.png", raw_candidates_vis)
            self._debug_save_json(
                run_dir,
                "dino_raw_candidates.json",
                {
                    "target": target,
                    "caption": enhanced_caption,
                    "num_candidates": len(raw_candidates),
                    "phrases": phrases,
                    "candidates": raw_candidates,
                },
            )
            
            if len(boxes) == 0:
                payload = {
                    "success": False,
                    "mode": "grounding_dino",
                    "target": target,
                    "enhanced_caption": enhanced_caption,
                    "error": f'Target "{target}" not detected in image',
                    "num_candidates": 0,
                }
                self._debug_save_json(run_dir, "result.json", payload)
                return {
                    'success': False,
                    'error': f'Target "{target}" not detected in image',
                    'debug_dir': str(run_dir) if run_dir is not None else None,
                }
            
            # Activate Siamese network: boost confidence using few-shot matching if available
            target_key = self._normalize_object_name(target)
            target_has_refs = (
                bool(self.few_shot_matcher)
                and bool(target_key)
                and target_key in self.few_shot_matcher.reference_db
            )
            logits = self.hybrid_detect_and_match(img_np, image_tensor, target, boxes, logits)
            if torch.is_tensor(logits):
                hybrid_scores = logits.detach().cpu().numpy().astype(np.float32)
            else:
                hybrid_scores = np.asarray(logits, dtype=np.float32)

            hybrid_candidates = []
            for idx, candidate in enumerate(raw_candidates):
                new_candidate = dict(candidate)
                if idx < len(hybrid_scores):
                    new_candidate["score"] = float(hybrid_scores[idx])
                new_candidate["label"] = (
                    f"{idx}: {float(new_candidate.get('score', 0.0)):.3f} "
                    f"(raw {float(candidate.get('score', 0.0)):.3f})"
                )
                hybrid_candidates.append(new_candidate)

            hybrid_candidates_vis = self._debug_draw_candidates(
                img_np,
                hybrid_candidates,
                color=(80, 255, 120),
                header_text=f"DINO + Siamese candidates: {len(hybrid_candidates)}",
            )
            self._debug_save_bgr_image(run_dir, "dino_candidates_hybrid.png", hybrid_candidates_vis)
            self._debug_save_json(
                run_dir,
                "dino_hybrid_candidates.json",
                {
                    "target": target,
                    "target_has_refs": bool(target_has_refs),
                    "num_candidates": len(hybrid_candidates),
                    "candidates": hybrid_candidates,
                },
            )

            if torch.is_tensor(logits):
                best_idx = int(torch.argmax(logits).item())
                best_score = float(logits[best_idx].item())
            else:
                logits_arr = np.asarray(logits, dtype=np.float32)
                best_idx = int(np.argmax(logits_arr))
                best_score = float(logits_arr[best_idx])

            logger.info("Post-Siamese best_idx=%d confidence=%.4f", best_idx, best_score)

            if target_has_refs:
                min_required_hybrid = float(os.getenv("HYBRID_TARGET_REQUIRED_SCORE", "0.35"))
                if best_score < min_required_hybrid:
                    payload = {
                        "success": False,
                        "mode": "grounding_dino",
                        "target": target,
                        "target_has_refs": True,
                        "best_score": float(best_score),
                        "required_score": float(min_required_hybrid),
                        "error": f'Target "{target}" not confidently matched from examples',
                    }
                    self._debug_save_json(run_dir, "result.json", payload)
                    return {
                        'success': False,
                        'error': f'Target "{target}" not confidently matched from examples',
                        'debug_dir': str(run_dir) if run_dir is not None else None,
                    }
            
            # Get bounding box
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
                self._debug_save_json(
                    run_dir,
                    "result.json",
                    {
                        "success": False,
                        "mode": "grounding_dino",
                        "target": target,
                        "error": f"Depth estimation error: {str(depth_error)}",
                    },
                )
                return {
                    'success': False,
                    'error': f'Depth estimation error: {str(depth_error)}',
                    'debug_dir': str(run_dir) if run_dir is not None else None,
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
            self._debug_save_bgr_image(run_dir, "selected_bbox.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'target': target,
                'angle': float(angle),
                'steps': float(steps),
                'distance_meters': float(meters),
                'depth': float(obj_depth),
                'bbox': [x1, y1, x2, y2],
                'visualization': img_base64,
                'confidence': best_score,
                'processing_time': float(processing_time),
                'surfaces': surfaces  # New: spatial relationship info
            }
            if run_dir is not None:
                result["debug_dir"] = str(run_dir)
                self._debug_save_json(
                    run_dir,
                    "result.json",
                    {
                        "mode": "grounding_dino",
                        "target": target,
                        "enhanced_caption": enhanced_caption,
                        "best_idx": int(best_idx),
                        "best_score": float(best_score),
                        "bbox": [x1, y1, x2, y2],
                        "result": result,
                    },
                )
            return result
        
        except Exception as e:
            print(f"[ERROR] process_image exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self._debug_save_json(
                run_dir,
                "result.json",
                {
                    "success": False,
                    "mode": "grounding_dino",
                    "target": target,
                    "error": f"Processing error: {str(e)}",
                    "exception_type": type(e).__name__,
                },
            )
            return {
                'success': False,
                'error': f'Processing error: {str(e)}',
                'debug_dir': str(run_dir) if run_dir is not None else None,
            }

