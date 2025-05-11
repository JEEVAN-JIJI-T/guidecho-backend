from flask import Blueprint, request, jsonify
from models.yolo_model import detect_objects
import base64
import cv2
import numpy as np

detect_bp = Blueprint('detect', __name__)

@detect_bp.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 to image
        image_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        label = detect_objects(img)
        return jsonify({'label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
