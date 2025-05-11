import torch
from pathlib import Path
import sys
import numpy as np
import cv2  # ✅ You forgot to import this
import os

# Dynamically download model if not present
MODEL_FILENAME = 'yolov5s.pt'
MODEL_URL = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'

# Get YOLOv5 repo path
yolov5_path = Path(__file__).parent.parent / 'yolov5'
sys.path.insert(0, str(yolov5_path))

# ✅ Download the model if not found
weights = yolov5_path / MODEL_FILENAME
if not weights.exists():
    import requests
    print(f"Downloading {MODEL_FILENAME}...")
    weights.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(MODEL_URL)
    with open(weights, 'wb') as f:
        f.write(r.content)

# YOLOv5 imports
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Set device
device = select_device('cpu')  # Use '0' for GPU locally
model = DetectMultiBackend(weights, device=device)
model.model.eval()

def detect_objects(img):
    # Preprocess image (OpenCV format)
    img = cv2.resize(img, (640, 640))
    img = img[..., ::-1]  # BGR to RGB
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process results
    labels = []
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                labels.append(model.names[int(cls)])
    return ', '.join(set(labels)) if labels else 'No obstacles detected'
