import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import random

def load_images_from_folder(folder_path, image_size=(224, 224)):
    """
    Load images from a folder and resize them
    
    Args:
        folder_path (str): Path to the folder containing images
        image_size (tuple): Target size for resizing images
    
    Returns:
        list: List of processed images
        list: List of corresponding filenames
    """
    images = []
    filenames = []
    
    # Supported image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    for extension in extensions:
        for img_path in glob.glob(os.path.join(folder_path, extension)):
            try:
                # Load image using PIL
                img = Image.open(img_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(image_size)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                images.append(img_array)
                filenames.append(os.path.basename(img_path))
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, filenames

def load_poultry_dataset(base_path, classes=['Salmonella', 'NewCastle', 'Coccidiosis', 'Healthy']):
    """
    Load poultry disease dataset with labels
    
    Args:
        base_path (str): Base directory containing class folders
        classes (list): List of disease class names
    
    Returns:
        numpy.array: Images array
        numpy.array: Labels array
        list: Class names
    """
    all_images = []
    all_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        
        if os.path.exists(class_path):
            print(f"Loading images from {class_name}...")
            images, _ = load_images_from_folder(class_path)
            
            # Add images and labels
            all_images.extend(images)
            all_labels.extend([class_idx] * len(images))
            
            print(f"Loaded {len(images)} images for {class_name}")
        else:
            print(f"Warning: Directory {class_path} not found!")
    
    return np.array(all_images), np.array(all_labels), classes

def display_sample_images(images, labels, class_names, num_samples=8):
    """
    Display sample images from the dataset
    
    Args:
        images (numpy.array): Array of images
        labels (numpy.array): Array of labels
        class_names (list): List of class names
        num_samples (int): Number of samples to display
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    # Randomly select samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(f'{class_names[labels[idx]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def preprocess_images(images):
    """
    Preprocess images for training
    
    Args:
        images (numpy.array): Array of images
    
    Returns:
        numpy.array: Preprocessed images
    """
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    
    return images

def create_train_test_split(images, labels, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    
    Args:
        images (numpy.array): Array of images
        labels (numpy.array): Array of labels
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(images, labels, test_size=test_size, 
                          random_state=random_state, stratify=labels)

# Example usage
if __name__ == "__main__":
    # Set your dataset path here
    dataset_path = "path/to/your/poultry_dataset"
    
    # Load the dataset
    print("Loading poultry disease dataset...")
    images, labels, class_names = load_poultry_dataset(dataset_path)
    
    print(f"\nDataset Summary:")
    print(f"Total images: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Classes: {class_names}")
    
    # Display class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"{class_names[class_idx]}: {count} images")
    
    # Preprocess images
    print("\nPreprocessing images...")
    images = preprocess_images(images)
    
    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = create_train_test_split(images, labels)
    
    print(f"Training set: {len(X_train)} images")
    print(f"Testing set: {len(X_test)} images")
    
    # Display sample images
    print("Displaying sample images...")
    display_sample_images(images * 255, labels, class_names)

# Alternative method using OpenCV
def load_images_opencv(folder_path, image_size=(224, 224)):
    """
    Alternative method to load images using OpenCV
    """
    images = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            
            # Load image using OpenCV
            img = cv2.imread(img_path)
            
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, image_size)
                
                images.append(img)
                filenames.append(filename)
    
    return images, filenames

# Function to load a single random image for testing
def load_random_image(folder_path):
    """
    Pick and load a random image from a folder
    """
    image_files = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for extension in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
    if image_files:
        random_image_path = random.choice(image_files)
        img = Image.open(random_image_path)
        
        # Convert to RGB and resize
        img = img.convert('RGB').resize((224, 224))
        
        return np.array(img), os.path.basename(random_image_path)
    else:
        print("No images found in the specified folder!")
        return None, None