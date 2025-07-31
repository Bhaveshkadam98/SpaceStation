import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="🛰️ OrbitalEye", layout="centered")

# Custom CSS with blur effect and dark theme
st.markdown("""
    <style>
    body {
        background: url('https://images.unsplash.com/photo-1582197297506-f3195cf6bc52?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
        background-size: cover;
    }

    .stApp {
        background-color: rgba(13, 17, 23, 0.85); /* translucent dark overlay */
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 2rem;
        border-radius: 12px;
        color: #ffffff;
    }

    h1, h4 {
        color: #00e0ff;
        text-align: center;
    }

    .stSlider > div > div {
        color: #ffffff;
    }

    .css-18e3th9 {
        background-color: rgba(13, 17, 23, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1>🛰️ OrbitalEye</h1>", unsafe_allow_html=True)
st.markdown("<h4>Real-Time Object Detection for Space Safety</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for upload and confidence threshold
with st.sidebar:
    st.header("📂 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    confidence_threshold = st.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.01)

# Load YOLO model
model = YOLO("best.pt")
class_names = model.names

# Display detection result
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    results = model.predict(source=img_np, conf=confidence_threshold, save=False)
    annotated_img = img_np.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names[cls]}: {conf:.2f}"

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(annotated_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)

    st.image(annotated_img, caption="🧠 Detected Objects", use_container_width=True)
else:
    st.info("👈 Upload an image from the sidebar to get started.")
