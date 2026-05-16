import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import time
from groundingdino.util.inference import load_model, predict, load_image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

device = "cpu"

st.title("GroundingDINO + Depth Anything V2 Navigation")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
target = st.text_input("Enter target object (e.g., chair, door, person)")

@st.cache_resource
def load_models():
    # GroundingDINO
    grounding_model = load_model(
        "GroundingDINO_SwinT_OGC.py",
        "groundingdino_swint_ogc.pth"
    )

    # Depth Anything V2
    depth_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-base-hf"
    )
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-base-hf"
    )

    return grounding_model, depth_processor, depth_model

grounding_model, depth_processor, depth_model = load_models()

if uploaded_file and target:
    # Load image properly for DINO
    image_source, image_tensor = load_image(uploaded_file)  # PIL image + tensor
    img_np = np.array(image_source)

    if st.button("Estimate"):
        start_time = time.time()
        # -------- GroundingDINO Detection --------
        boxes, logits, phrases = predict(
            model=grounding_model,
            image=image_tensor,  # tensor, not np.array
            caption=target,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )

        if len(boxes) == 0:
            st.error("Target not detected")
            st.stop()

        h, w, _ = img_np.shape
        box = boxes[0] * torch.tensor([w, h, w, h])
        cx, cy, bw, bh = box
        x1 = int(cx - bw/2)
        y1 = int(cy - bh/2)
        x2 = int(cx + bw/2)
        y2 = int(cy + bh/2)

        # Draw bounding box
        vis = img_np.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(vis, caption="Detected Target")

        # -------- Depth Anything V2 --------
        with torch.no_grad():
            inputs = depth_processor(images=image_source, return_tensors="pt")
            outputs = depth_model(**inputs)
            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth_map = depth.detach().cpu().numpy()
        obj_depth = depth_map[y1:y2, x1:x2].mean()

        # -------- Angle Calculation --------
        img_center = w / 2
        obj_center = (x1 + x2) / 2
        fov = 60
        angle = (obj_center - img_center) / w * fov

        # -------- Steps --------
        meters = obj_depth * 0.6
        steps = meters / 0.75

        end_time = time.time()
        elapsed = end_time - start_time

        st.info(f"Time taken for estimation: {elapsed:.2f} seconds")
        
        
        st.success(f"Angle: {angle:.2f}°")
        st.success(f"Estimated steps: {steps:.1f}")