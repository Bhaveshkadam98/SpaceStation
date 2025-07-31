# 🚀 AstroGuard - Object Detection for Space Safety

AstroGuard is a lightweight YOLOv8-based object detection model built to identify critical tools in space station environments — like oxygen tanks, fire extinguishers, and toolboxes — using synthetic training data.

### 🛰️ Features
- Detects key onboard tools in images
- Trained on Falcon's synthetic dataset
- Streamlit app for interactive testing
- mAP@0.5: **94.3%**

### 📦 Contents
- `best.pt` - Trained YOLOv8 model
- `app.py` - Streamlit web app
- `yolo_params.yaml` - Dataset config
- `runs/` - Training logs and results

### 🚀 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
