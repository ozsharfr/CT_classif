"""
Configuration Module

Centralized configuration for all CT brain classification parameters.
Edit this file to adjust paths, hyperparameters, and preprocessing settings.
"""

import torch

# ==============================
# Paths
# ==============================
DICOM_DIR = "C:/Users/USER/Downloads/CTs-20260215T111200Z-1-001/CTs"
OUTPUT_DIR = "./outputs"
LABELS_CSV = "C:/Users/USER/CT/labels.csv"
EDA_CSV = "./outputs/eda_agglomerative.csv"
GRADCAM_DIR = "./outputs/gradcam"

# ==============================
# Brain Window (NCCT Standard)
# ==============================
WL = 0.0  # Window level
WW = 1e6   # Window width- set to 80, since a very large number caused poor contrast
WMIN = WL - WW/2
WMAX = WL + WW/2

# ==============================
# Image Preprocessing
# ==============================
FRAC_CROP = 0.9  # Center crop fraction (0.75-0.95)
INPUT_SIZE = 224  # Model input size (128/224/256/384/512)

# ==============================
# Early Stopping
# ==============================
EARLY_STOPPING_PATIENCE = 5  # Stop after N epochs without improvement
EARLY_STOPPING_METRIC = 'auc'  # Metric to monitor: 'auc', 'loss', 'acc'
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# ==============================
# Model Checkpointing
# ==============================
MODEL_CHECKPOINT_DIR = "./outputs/models"
SAVE_BEST_MODEL = True  # Save model with best validation performance
SAVE_FINAL_MODEL = True  # Save model after training completes
SAVE_TRAINING_HISTORY = True  # Save training metrics history

# ==============================
# Data Augmentationf
# ==============================
# Toggle augmentation on/off
USE_AUGMENTATION = True  # Set to False to disable all augmentation

# Geometric augmentation parameters (only used if USE_AUGMENTATION = True)
# Note: Only geometric augmentations for CT - NO brightness/contrast
# (Brightness/contrast would corrupt Hounsfield Unit physical values)
AUG_ROTATION_DEGREES = 10  # Random rotation ±10°
AUG_TRANSLATE = (0.05, 0.05)  # Random translation ±5%
AUG_SCALE = (0.95, 1.05)  # Random scale 95-105%
AUG_HORIZONTAL_FLIP = False  # Random horizontal flip

# ==============================
# Normalization Configuration
# ==============================
# How normalization is determined:
# - If USE_PRETRAINED = True: MUST use pretrained model's normalization (ImageNet)
# - If USE_PRETRAINED = False: CAN compute from your dataset
COMPUTE_NORMALIZATION_FROM_DATASET = False  # Only set True if USE_PRETRAINED = False

# ==============================
# Model Configuration
# ==============================
MODEL_NAME = 'resnet18'  # Model architecture
USE_PRETRAINED = True  # Use ImageNet pretrained weights


# ImageNet normalization statistics (REQUIRED for pretrained models)
# Source: PyTorch official documentation
# https://pytorch.org/vision/stable/models.html
# 
# These values were computed from 1.2M ImageNet training images.
# They are mathematically required for pretrained weights to work correctly.
# The first convolutional layer weights expect inputs in this distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Verify against PyTorch's documented values
assert IMAGENET_MEAN == [0.485, 0.456, 0.406], "ImageNet mean mismatch!"
assert IMAGENET_STD == [0.229, 0.224, 0.225], "ImageNet std mismatch!"

# Verify configuration consistency
if USE_PRETRAINED and COMPUTE_NORMALIZATION_FROM_DATASET:
    raise ValueError(
        "Cannot compute normalization from dataset when using pretrained weights! "
        "Pretrained models require their original normalization statistics. "
        "Set COMPUTE_NORMALIZATION_FROM_DATASET = False"
    )

# Dataset statistics (will be computed if COMPUTE_NORMALIZATION_FROM_DATASET = True)
DATASET_MEAN = None
DATASET_STD = None


# ==============================
# Training Hyperparameters
# ==============================
BATCH_SIZE = 32  # Reduce if GPU memory error
MAX_EPOCHS = 50  # Maximum training epochs (early stopping may stop before)
LR = 1e-4  # Learning rate
SEED = 42  # Random seed for reproducibility
FREEZE_LAYERS = False  # Freeze all layers except final fully connected layer

# Train/Val/Test split
VALIDATION_SIZE = 0.2  # 20% for validation
TEST_SIZE = 0.2  # 20% for test (remaining 60% for training)

# ==============================
# Clustering Parameters
# ==============================
CLUST_THRESH = 5  # Clustering threshold percentile (0-100)
MIN_CLUST_SIZE = 3  # Minimum cluster size for reporting
NON_AIR = -500  # HU threshold to separate air from tissue

# ==============================
# Explainability Parameters
# ==============================
GRADCAM_TARGET_LAYER = "layer4"  # ResNet layer for Grad-CAM
GRADCAM_MAX_PER_CLUSTER = 6  # Max images per cluster
GRADCAM_MIN_CLUSTER_SIZE = 4  # Min cluster size for visualization
GRADCAM_ALPHA = 0.45  # Heatmap transparency (0-1)

# ==============================
# Device Configuration
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)
