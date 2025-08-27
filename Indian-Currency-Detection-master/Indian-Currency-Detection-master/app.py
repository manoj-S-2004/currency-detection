from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# Load the trained model
model = tf.keras.models.load_model('Model_training/currency_model.h5')

# Load class labels
with open('Model_training/class_names.txt', 'r') as f:
    class_names = f.read().splitlines()

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 100, 100, 1).astype('float32') / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' in request.files:
        image = request.files['image']
        image_path = os.path.join('static', image.filename)
        image.save(image_path)

        img = cv2.imread(image_path)
        processed = preprocess_image(img)
        pred = model.predict(processed)[0]
        idx = np.argmax(pred)
        label = class_names[idx]
        confidence = pred[idx] * 100

        return render_template('result.html', label=label, confidence=confidence, image_path=image_path)
    return 'No image uploaded', 400

@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed = preprocess_image(frame)
    pred = model.predict(processed)[0]
    idx = np.argmax(pred)
    label = class_names[idx]
    confidence = float(pred[idx]) * 100

    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
