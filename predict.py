import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('poultry_disease_model_dummy.h5')

# Create a fake test image (224x224 RGB)
test_image = np.random.rand(1, 224, 224, 3).astype('float32')

# Make prediction
prediction = model.predict(test_image)

# Get class labels
class_names = ['Healthy', 'Coccidiosis', 'Salmonella', 'NewCastle']

# Show results
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"🧠 Predicted Class: {predicted_class} ({confidence * 100:.2f}%)")
