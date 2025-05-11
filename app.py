from flask import Flask
from flask_cors import CORS
from routes.detect import detect_bp
import os

app = Flask(__name__)
CORS(app)
app.register_blueprint(detect_bp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # required by Render
    app.run(host='0.0.0.0', port=port)
