"""
Configuration file for Food Classification project
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'fruits-360-original-size-main')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
VAL_DIR = os.path.join(DATA_DIR, 'Validation')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configurations
IMAGE_SIZE = (224, 224)  # Standard size for MobileNet/ResNet
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model selection: 'mobilenet' or 'resnet50'
MODEL_TYPE = 'mobilenet'  # Change to 'resnet50' if needed

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2

# OpenCV preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Validation split
VALIDATION_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42
