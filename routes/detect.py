from flask import Blueprint, request, jsonify
from models.yolo_model import detect_objects
import base64
import cv2
import numpy as np

detect_bp = Blueprint('detect', __name__)

@detect_bp.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    label = detect_objects(img)
    return jsonify({ 'label': label })