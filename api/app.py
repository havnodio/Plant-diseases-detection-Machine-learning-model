from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # ← ADD THIS
from PIL import Image
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load model & classes
model = tf.keras.models.load_model('../model/plant_disease_model.keras')
with open('../model/class_names.json') as f:
    class_names = json.load(f)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Use EfficientNet's preprocessing (SAME as training!)
    return preprocess_input(img_array)  # ← CHANGED THIS LINE

@app.route('/')
def home():
    return '<h1>Plant Disease API Running ✅</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    processed = preprocess_image(img)
    pred = model.predict(processed)[0]
    conf = float(np.max(pred))
    disease = class_names[np.argmax(pred)]
    
    return jsonify({
        'disease': disease,
        'confidence': round(conf * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True,port=5001)
