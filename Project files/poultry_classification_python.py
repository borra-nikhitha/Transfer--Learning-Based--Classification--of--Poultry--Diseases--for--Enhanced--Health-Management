# Complete Python Code for Poultry Disease Classification using Transfer Learning
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class PoultryDiseaseClassifier:
    def __init__(self, image_size=(224, 224), batch_size=32):
        """
        Initialize the Poultry Disease Classifier
        
        Args:
            image_size (tuple): Size to resize images to
            batch_size (int): Batch size for training
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes = ['Healthy', 'Salmonella', 'Newcastle', 'Coccidiosis']
        self.num_classes = len(self.classes)
        
    def create_sample_data(self, num_samples_per_class=100):
        """
        Create sample synthetic data for demonstration
        """
        print("Creating synthetic sample data...")
        X_data = []
        y_data = []
        
        for class_idx, class_name in enumerate(self.classes):
            print(f"Generating {num_samples_per_class} samples for {class_name}...")
            
            for i in range(num_samples_per_class):
                # Create synthetic image data with class-specific patterns
                if class_name == 'Healthy':
                    # Healthy birds - brighter, more uniform patterns
                    img = np.random.randint(120, 200, (*self.image_size, 3), dtype=np.uint8)
                    # Add some texture
                    noise = np.random.normal(0, 10, (*self.image_size, 3))
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    
                elif class_name == 'Salmonella':
                    # Salmonella - reddish tints, more irregular patterns
                    img = np.random.randint(80, 160, (*self.image_size, 3), dtype=np.uint8)
                    img[:, :, 0] += np.random.randint(20, 40)  # Add red tint
                    noise = np.random.normal(0, 20, (*self.image_size, 3))
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    
                elif class_name == 'Newcastle':
                    # Newcastle Disease - yellowish/orange tints
                    img = np.random.randint(90, 170, (*self.image_size, 3), dtype=np.uint8)
                    img[:, :, 0] += np.random.randint(15, 30)  # Red
                    img[:, :, 1] += np.random.randint(15, 30)  # Green (red+green=yellow)
                    noise = np.random.normal(0, 15, (*self.image_size, 3))
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    
                else:  # Coccidiosis
                    # Coccidiosis - darker, more muted colors
                    img = np.random.randint(60, 140, (*self.image_size, 3), dtype=np.uint8)
                    img[:, :, 2] += np.random.randint(10, 25)  # Add slight blue tint
                    noise = np.random.normal(0, 25, (*self.image_size, 3))
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
                X_data.append(img)
                y_data.append(class_idx)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Created dataset with shape: {X_data.shape}")
        return X_data, y_data
    
    def load_real_dataset(self, dataset_path):
        """
        Load real dataset from folder structure
        
        Args:
            dataset_path (str): Path to dataset with class folders
        
        Returns:
            X_data, y_data: Images and labels
        """
        X_data = []
        y_data = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(dataset_path, class_name)
            
            if os.path.exists(class_path):
                print(f"Loading images from {class_name}...")
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                for img_file in image_files:
                    try:
                        img_path = os.path.join(class_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.image_size)
                        img_array = np.array(img)
                        
                        X_data.append(img_array)
                        y_data.append(class_idx)
                        
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
                
                print(f"Loaded {len([f for f in image_files])} images from {class_name}")
            else:
                print(f"Warning: Directory {class_path} not found!")
        
        return np.array(X_data), np.array(y_data)
    
    def preprocess_data(self, X_data, y_data, test_size=0.2):
        """
        Preprocess the data for training
        
        Args:
            X_data: Image data
            y_data: Labels
            test_size: Proportion for test set
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Preprocessing data...")
        
        # Normalize pixel values
        X_data = X_data.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_data = to_categorical(y_data, num_classes=self.num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=42, stratify=y_data
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_generators(self, X_train, y_train, X_val=None, y_val=None):
        """
        Create data generators with augmentation
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train, batch_size=self.batch_size
        )
        
        if X_val is not None:
            val_generator = val_datagen.flow(
                X_val, y_val, batch_size=self.batch_size
            )
            return train_generator, val_generator
        
        return train_generator
    
    def build_transfer_learning_model(self, base_model_name='ResNet50', trainable_layers=0):
        """
        Build transfer learning model
        
        Args:
            base_model_name: Name of base model ('ResNet50', 'VGG16', 'MobileNetV2')
            trainable_layers: Number of top layers to make trainable
        """
        print(f"Building {base_model_name} transfer learning model...")
        
        # Load pre-trained base model
        if base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                input_shape=(*self.image_size, 3))
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                             input_shape=(*self.image_size, 3))
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                   input_shape=(*self.image_size, 3))
        else:
            raise ValueError("Unsupported base model")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make some top layers trainable for fine-tuning
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Add custom classification layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully!")
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train the model with callbacks
        """
        print("Starting model training...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_poultry_model.h5', monitor='val_accuracy', 
                          save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        ]
        
        # Calculate steps
        steps_per_epoch = len(X_train) // self.batch_size
        validation_steps = len(X_val) // self.batch_size
        
        # Train model
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=self.classes)
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        return accuracy, report, cm
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix - Poultry Disease Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_disease(self, image_path_or_array, return_probabilities=False):
        """
        Predict disease from image
        
        Args:
            image_path_or_array: Path to image file or numpy array
            return_prob