import torch
from pathlib import Path
import sys

# Add YOLOv5 repo to path
yolov5_path = Path(__file__).parent / 'yolov5'
sys.path.insert(0, str(yolov5_path))

from yolov5.models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# Load model
weights = yolov5_path / 'yolov5s.pt'
device = select_device('cpu')  # Change to '0' if using GPU
model = DetectMultiBackend(weights, device=device)
model.model.eval()

def detect_objects(img):
    # Convert image (numpy) to tensor
    img_tensor = torch.from_numpy(img).to(device)
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
