import tensorflow as tf
import numpy as np

# Create fake image data: 100 samples of 224x224 RGB images
num_classes = 4
num_samples = 100
img_size = 224

# Random image data (100 images, 224x224, 3 channels)
x_train = np.random.rand(num_samples, img_size, img_size, 3).astype('float32')
y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)

# Random validation data
x_val = np.random.rand(20, img_size, img_size, 3).astype('float32')
y_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 20), num_classes)

# Build model using transfer learning
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model on fake data
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3)

# Save model
model.save("poultry_disease_model_dummy.h5")

print("✅ Training complete (no real images used). Model saved as poultry_disease_model_dummy.h5")

