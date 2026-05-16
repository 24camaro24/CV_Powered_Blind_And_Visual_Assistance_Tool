import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("Depth Navigation Estimator")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
target = st.text_input("Enter target object")

# Load models once
@st.cache_resource
def load_models():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    depth_processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    return processor, detector, depth_processor, depth_model

processor, detector, depth_processor, depth_model = load_models()

if uploaded_file and target:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    if st.button("Estimate"):

        # ---------- OBJECT DETECTION ----------
        inputs = processor(text=[target], images=image, return_tensors="pt").to(device)
        outputs = detector(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]

        if len(results["boxes"]) == 0:
            st.error("Target not detected")
        else:
            #box = results["boxes"][0].cpu().numpy()
            box = results["boxes"][0].detach().cpu().numpy()

            x1,y1,x2,y2 = map(int, box)

            # ---------- DEPTH ----------
            with torch.no_grad():
                depth_inputs = depth_processor(images=image, return_tensors="pt").to(device)
                depth = depth_model(**depth_inputs).predicted_depth

            #depth_inputs = depth_processor(images=image, return_tensors="pt").to(device)
            #depth = depth_model(**depth_inputs).predicted_depth

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = depth.cpu().numpy()

            obj_depth = depth_map[y1:y2, x1:x2].mean()

            # ---------- ANGLE ----------
            img_center = image.size[0] / 2
            obj_center = (x1 + x2) / 2
            fov = 60   # assume camera FOV
            angle = (obj_center - img_center) / image.size[0] * fov

            # ---------- STEPS ----------
            meters = float(obj_depth / 1000 * 3)   # rough scaling
            steps = meters / 0.75

            st.image(image, caption="Uploaded")
            st.success(f"Angle: {angle:.2f} degrees")
            st.success(f"Estimated Steps: {steps:.1f}")
