import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def detect_objects(img):
    results = model(img)
    labels = results.pandas().xyxy[0]['name'].tolist()
    return ', '.join(set(labels)) if labels else 'No obstacles detected'