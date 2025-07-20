from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model  

from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.environ.get("MODEL_PATH", "models/poultry_disease_model.h5")
print(f"Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define class labels
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # adjust size if needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def home():
    return "✔️ Poultry Disease Classifier Running"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filepath = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        os.remove(filepath)

        return jsonify({
            'predicted_disease': predicted_class,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({'error': f"Exception during prediction: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
