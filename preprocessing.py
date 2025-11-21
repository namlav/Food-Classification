"""
Image preprocessing module using OpenCV
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import config


class ImagePreprocessor:
    """
    Image preprocessing class with OpenCV techniques
    """
    
    def __init__(self, target_size: Tuple[int, int] = config.IMAGE_SIZE):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_SIZE
        )
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising using Non-local Means Denoising
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def adjust_brightness_contrast(
        self, 
        image: np.ndarray, 
        brightness: float = 1.0, 
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
        return adjusted
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(
        self, 
        image: np.ndarray, 
        apply_clahe: bool = True,
        apply_denoise: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (BGR format from cv2.imread)
            apply_clahe: Whether to apply CLAHE enhancement
            apply_denoise: Whether to apply denoising
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed image in RGB format
        """
        # Resize
        processed = self.resize_image(image)
        
        # Apply CLAHE if requested
        if apply_clahe:
            processed = self.apply_clahe(processed)
        
        # Apply denoising if requested
        if apply_denoise:
            processed = self.denoise(processed)
        
        # Convert BGR to RGB (for TensorFlow/Keras)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize if requested
        if normalize:
            processed = self.normalize(processed)
        
        return processed
    
    def preprocess_from_path(
        self, 
        image_path: str,
        apply_clahe: bool = True,
        apply_denoise: bool = False,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to image file
            apply_clahe: Whether to apply CLAHE enhancement
            apply_denoise: Whether to apply denoising
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed image or None if loading fails
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        return self.preprocess(image, apply_clahe, apply_denoise, normalize)


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentation to image using OpenCV
    
    Args:
        image: Input image (RGB, normalized)
        
    Returns:
        Augmented image
    """
    # Convert back to uint8 for OpenCV operations
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-config.ROTATION_RANGE, config.ROTATION_RANGE)
        h, w = img_uint8.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_uint8 = cv2.warpAffine(img_uint8, M, (w, h))
    
    # Random horizontal flip
    if config.HORIZONTAL_FLIP and np.random.random() > 0.5:
        img_uint8 = cv2.flip(img_uint8, 1)
    
    # Random brightness/contrast
    brightness = np.random.uniform(*config.BRIGHTNESS_RANGE)
    contrast = np.random.uniform(*config.CONTRAST_RANGE)
    img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=contrast, beta=(brightness - 1) * 50)
    
    # Convert back to float32 normalized
    return img_uint8.astype(np.float32) / 255.0


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    print("ImagePreprocessor initialized successfully")
    print(f"Target size: {preprocessor.target_size}")
