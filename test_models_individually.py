#!/usr/bin/env python3
"""Test which model is causing the hang"""

import os
import sys
import signal

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

sys.path.insert(0, 'backend')

def timeout_handler(signum, frame):
    raise TimeoutError("Model loading timed out")

signal.signal(signal.SIGALRM, timeout_handler)

try:
    print('[1] Setting up CUDA patches...')
    signal.alarm(5)
    import torch
    signal.alarm(0)
    
    print('[2] Importing pipeline components...')
    signal.alarm(10)
    from pathlib import Path
    from groundingdino.util.inference import load_model
    signal.alarm(0)
    
    print('[3] Loading GroundingDINO...')
    signal.alarm(30)
    base_dir = Path('.')
    config_path = str(base_dir / "GroundingDINO_SwinT_OGC.py")
    model_path = str(base_dir / "groundingdino_swint_ogc.pth")
    model = load_model(config_path, model_path)
    signal.alarm(0)
    print('[✓] GroundingDINO loaded')
    
    print('[4] Loading Depth Anything...')
    signal.alarm(30)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-base-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-base-hf")
    signal.alarm(0)
    print('[✓] Depth model loaded')
    
    print('[5] Loading Whisper...')
    signal.alarm(30)
    from faster_whisper import WhisperModel
    whisper = WhisperModel("small", device='cpu')
    signal.alarm(0)
    print('[✓] Whisper loaded')
    
    print('[6] Loading Flan-T5...')
    signal.alarm(30)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    signal.alarm(0)
    print('[✓] Flan-T5 loaded')
    
    print('[SUCCESS] All models loaded!')
    
except TimeoutError as e:
    print(f'[TIMEOUT] {e}')
    sys.exit(1)
except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)
