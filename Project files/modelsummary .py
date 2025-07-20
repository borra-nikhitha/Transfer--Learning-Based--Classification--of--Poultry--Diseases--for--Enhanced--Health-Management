# Poultry Disease Classification Model - Transfer Learning Approach
# This code creates a model for classifying: Salmonella, Newcastle Disease, Coccidiosis, Healthy

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("="*60)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 4  # Salmonella, Newcastle Disease, Coccidiosis, Healthy

print("Building Transfer Learning Model for Poultry Disease Classification")
print("="*60)

# Load pre-trained ResNet50 model (without top layers)
base_model = ResNet50(
    weights='imagenet',  # Use ImageNet pre-trained weights
    include_top=False,   # Don't include the final classification layer
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze the base model layers (for transfer learning)
base_model.trainable = False

print("Base Model: ResNet50 (Pre-trained on ImageNet)")
print(f"Input Shape: ({IMG_HEIGHT}, {IMG_WIDTH}, 3)")
print(f"Number of Classes: {NUM_CLASSES}")
print("="*60)

# Add custom classification layers on top
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)  # Base model
x = GlobalAveragePooling2D()(x)         # Global average pooling
x = Dropout(0.3)(x)                     # Dropout for regularization
x = Dense(128, activation='relu')(x)     # Dense layer with ReLU
x = Dropout(0.2)(x)                     # Another dropout
outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)  # Final classification layer

# Create the complete model
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("MODEL ARCHITECTURE SUMMARY")
print("="*60)
print("Disease Classes:")
print("1. Salmonella")
print("2. Newcastle Disease") 
print("3. Coccidiosis")
print("4. Healthy")
print("="*60)

# Display the model summary - THIS IS WHAT YOU NEED TO SCREENSHOT
model.summary()

print("="*60)
print("MODEL DETAILS:")
print(f"Total Parameters: {model.count_params():,}")
print(f"Trainable Parameters: {sum([tf.keras.utils.get_weight_path(w).shape.as_list() for w in model.trainable_weights])}")
print(f"Non-trainable Parameters: {base_model.count_params():,}")

print("="*60)
print("TRANSFER LEARNING APPROACH:")
print("✓ Using ResNet50 pre-trained on ImageNet")
print("✓ Frozen base model layers (transfer learning)")
print("✓ Custom classification head for 4 poultry diseases")
print("✓ GlobalAveragePooling + Dense layers")
print("✓ Dropout for regularization")

print("="*60)
print("🔍 SCREENSHOT THIS ENTIRE OUTPUT FOR YOUR DOCUMENT")
print("="*60)

# Optional: Create a visualization of the model architecture
try:
    tf.keras.utils.plot_model(
        model,
        to_file='poultry_disease_model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=150
    )
    print("📊 Model architecture diagram saved as 'poultry_disease_model_architecture.png'")
except:
    print("📊 Model visualization not available (graphviz not installed)")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Screenshot this model summary output")
print("2. Use this for your Model Performance Test document")
print("3. Train this model with your poultry disease dataset")
print("4. Capture training accuracy screenshots")
print("="*60)