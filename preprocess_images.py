import os
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import pandas as pd

def preprocess_images(input_dir, output_dir, target_size=(800, 640)):
    """
    Preprocess all images in the input directory by resizing them to target_size
    and saving them to the output directory.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save preprocessed images
        target_size: Tuple of (height, width) for target image size
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to preprocess")
    print(f"Target size: {target_size}")
    
    # Process each image
    for filename in tqdm(image_files, desc="Preprocessing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load image
            image = io.imread(input_path)
            
            # Resize image
            resized_image = transform.resize(image, target_size, mode='constant', anti_aliasing=True)
            
            # Convert to uint8 if needed (for bmp files)
            if filename.lower().endswith('.bmp'):
                resized_image = (resized_image * 255).astype(np.uint8)
            
            # Save resized image
            io.imsave(output_path, resized_image)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Preprocessing complete! Resized images saved to {output_dir}")

def main():
    """Preprocess images for both training and test datasets"""
    
    # Configuration
    target_size = (800, 640)  # Same as used in training
    
    # Directories
    base_dir = "process_data"
    
    # Training data
    train_input = os.path.join(base_dir, "TrainingData")
    train_output = os.path.join(base_dir, "TrainingData_preprocessed")
    
    # Test data
    test1_input = os.path.join(base_dir, "Test1Data") 
    test1_output = os.path.join(base_dir, "Test1Data_preprocessed")
    
    test2_input = os.path.join(base_dir, "Test2Data")
    test2_output = os.path.join(base_dir, "Test2Data_preprocessed")
    
    print("Starting image preprocessing...")
    print("=" * 50)
    
    # Preprocess training data
    if os.path.exists(train_input):
        print(f"\nPreprocessing training data...")
        preprocess_images(train_input, train_output, target_size)
    else:
        print(f"Training data directory not found: {train_input}")
    
    # Preprocess test1 data
    if os.path.exists(test1_input):
        print(f"\nPreprocessing test1 data...")
        preprocess_images(test1_input, test1_output, target_size)
    else:
        print(f"Test1 data directory not found: {test1_input}")
    
    # Preprocess test2 data
    if os.path.exists(test2_input):
        print(f"\nPreprocessing test2 data...")
        preprocess_images(test2_input, test2_output, target_size)
    else:
        print(f"Test2 data directory not found: {test2_input}")
    
    print("\n" + "=" * 50)
    print("Image preprocessing complete!")

if __name__ == "__main__":
    main() 