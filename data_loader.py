"""
Data loading and preparation module
"""
import os
import numpy as np
import cv2
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import config
from preprocessing import ImagePreprocessor


class DataLoader:
    """
    Data loader for food classification dataset
    """
    
    def __init__(self, data_dir: str = config.DATA_DIR):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing data
        """
        self.data_dir = data_dir
        self.preprocessor = ImagePreprocessor()
        self.class_names = []
        self.num_classes = 0
        
    def load_dataset_from_directory(
        self, 
        directory: str,
        apply_clahe: bool = True,
        apply_denoise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dataset from directory structure:
        directory/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image1.jpg
                image2.jpg
        
        Args:
            directory: Root directory containing class folders
            apply_clahe: Whether to apply CLAHE preprocessing
            apply_denoise: Whether to apply denoising
            
        Returns:
            Tuple of (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        # Get all class directories
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return np.array([]), np.array([]), []
        
        class_dirs = sorted([d for d in os.listdir(directory) 
                           if os.path.isdir(os.path.join(directory, d))])
        
        if not class_dirs:
            print(f"Warning: No class directories found in {directory}")
            return np.array([]), np.array([]), []
        
        class_names = class_dirs
        
        print(f"Loading dataset from {directory}")
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Load images from each class
        for class_idx, class_name in enumerate(tqdm(class_names, desc="Loading classes")):
            class_path = os.path.join(directory, class_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                # Preprocess image
                processed_img = self.preprocessor.preprocess_from_path(
                    img_path,
                    apply_clahe=apply_clahe,
                    apply_denoise=apply_denoise
                )
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(class_idx)
        
        print(f"Loaded {len(images)} images from {len(class_names)} classes")
        
        return np.array(images), np.array(labels), class_names
    
    def prepare_data(
        self,
        train_dir: str = None,
        val_dir: str = None,
        test_split: float = 0.2,
        random_state: int = config.RANDOM_SEED
    ) -> Dict:
        """
        Prepare training and validation data
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory (optional)
            test_split: Fraction of data to use for validation if val_dir not provided
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/val data and metadata
        """
        if train_dir is None:
            train_dir = config.TRAIN_DIR
        
        # Load training data
        X_train, y_train, class_names = self.load_dataset_from_directory(train_dir)
        
        if len(X_train) == 0:
            raise ValueError(f"No training data found in {train_dir}")
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Load validation data if directory provided
        if val_dir and os.path.exists(val_dir):
            X_val, y_val, _ = self.load_dataset_from_directory(val_dir)
        else:
            # Split training data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=test_split,
                random_state=random_state,
                stratify=y_train
            )
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        print(f"\nDataset Summary:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Image shape: {X_train[0].shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train_cat,
            'X_val': X_val,
            'y_val': y_val_cat,
            'class_names': class_names,
            'num_classes': self.num_classes
        }


def download_fruits360_sample():
    """
    Instructions for downloading Fruits-360 dataset
    """
    instructions = """
    To download the Fruits-360 dataset:
    
    1. Visit: https://www.kaggle.com/datasets/moltean/fruits
    2. Download the dataset
    3. Extract to the 'data' folder with structure:
       data/
         train/
           Apple/
           Banana/
           ...
         test/
           Apple/
           Banana/
           ...
    
    Or use Kaggle API:
    kaggle datasets download -d moltean/fruits
    """
    print(instructions)
    return instructions


if __name__ == "__main__":
    # Print download instructions
    download_fruits360_sample()
    
    # Test data loader
    loader = DataLoader()
    print("\nDataLoader initialized successfully")
