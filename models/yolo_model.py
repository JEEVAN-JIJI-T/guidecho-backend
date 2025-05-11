import torch
from pathlib import Path
import sys
import numpy as np

# Add YOLOv5 repo to path
yolov5_path = Path(__file__).parent / 'yolov5'
sys.path.insert(0, str(yolov5_path))

# Import YOLOv5 utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load model
weights = yolov5_path / 'yolov5s.pt'
device = select_device('cpu')  # Use '0' if you want GPU on local
model = DetectMultiBackend(weights, device=device)
model.model.eval()

def detect_objects(img):
    # Convert image (OpenCV numpy format) to tensor
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

