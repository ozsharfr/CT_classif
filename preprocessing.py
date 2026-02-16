"""
Image Preprocessing Module

Contains all image preprocessing pipelines for different use cases:
- Model training/inference
- Visualization/display
- Embedding extraction
"""

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import pydicom

from config import (
    FRAC_CROP, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    USE_AUGMENTATION, AUG_ROTATION_DEGREES, AUG_TRANSLATE, 
    AUG_SCALE, AUG_HORIZONTAL_FLIP,
    USE_PRETRAINED, COMPUTE_NORMALIZATION_FROM_DATASET,
    DATASET_MEAN, DATASET_STD
)
from utils import dicom_to_hu, win01, apply_center_crop
import logging

logger = logging.getLogger(__name__)


def compute_dataset_statistics(dataloader):
    """
    Compute mean and std from your dataset.
    
    WARNING: Only use this if training from scratch (USE_PRETRAINED = False).
    For pretrained models, you MUST use their original normalization statistics.
    
    Args:
        dataloader: DataLoader with images in [0,1] range (before normalization)
    
    Returns:
        tuple: (mean, std) as lists of 3 floats (RGB channels)
    """
    logger.info("Computing normalization statistics from dataset...")
    logger.warning("This should ONLY be used when training from scratch!")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # First pass: compute mean
    for images, _, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    
    # Second pass: compute std
    for images, _, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0, 2])
    
    std = torch.sqrt(std / (total_samples * images.size(-1)))
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    logger.info(f"Computed dataset mean: {mean_list}")
    logger.info(f"Computed dataset std: {std_list}")
    
    return mean_list, std_list


def get_augmentation_transform(training=True):
    """
    Get geometric augmentation pipeline for training.
    
    Only includes geometric transformations (rotation, translation, scale, flip).
    Does NOT include intensity augmentations (brightness, contrast) as these would
    corrupt the physical meaning of Hounsfield Units in CT images.
    
    Args:
        training: If False, returns None (no augmentation)
    
    Returns:
        torchvision.transforms.Compose or None
    """
    if not training or not USE_AUGMENTATION:
        return None
    
    transforms_list = []
    
    # Geometric augmentations
    transforms_list.append(
        T.RandomAffine(
            degrees=AUG_ROTATION_DEGREES,
            translate=AUG_TRANSLATE,
            scale=AUG_SCALE
        )
    )
    
    # Horizontal flip
    if AUG_HORIZONTAL_FLIP:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
    
    return T.Compose(transforms_list)


def get_normalization_transform():
    """
    Get the appropriate normalization transform based on configuration.
    
    Returns:
        torchvision.transforms.Normalize or identity transform
    """
    if USE_PRETRAINED:
        # Using pretrained weights - MUST use their normalization
        logger.debug(f"Using ImageNet normalization (mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
        return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    elif COMPUTE_NORMALIZATION_FROM_DATASET:
        # Training from scratch - use computed dataset statistics
        if DATASET_MEAN is None or DATASET_STD is None:
            logger.warning("Dataset statistics not yet computed - using temporary normalization")
            return T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            logger.debug(f"Using dataset normalization (mean={DATASET_MEAN}, std={DATASET_STD})")
            return T.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    
    else:
        # No normalization
        logger.debug("No normalization applied")
        return T.Compose([])  # Identity transform


def load_dicom_as_tensor(path, out_size=INPUT_SIZE, return_1ch=False):
    """
    Load DICOM as 3-channel normalized tensor for model input.
    
    Complete preprocessing pipeline:
    1. DICOM → HU conversion
    2. Brain windowing to [0,1]
    3. Center crop (remove borders)
    4. Resize to fixed size
    5. Convert to 3-channel
    6. Apply ImageNet normalization (CRITICAL FIX!)
    
    Args:
        path: Path to DICOM file
        out_size: Output size (default from config)
        return_1ch: If True, also return single-channel for visualization
    
    Returns:
        x3: (3, H, W) tensor with normalization applied
        x_display: (H, W) array in [0,1] if return_1ch=True
    """
    ds = pydicom.dcmread(path, force=True)
    hu = dicom_to_hu(ds)
    x = win01(hu)

    # Center crop
    h, w = x.shape
    frac = FRAC_CROP
    ch, cw = int(h*frac), int(w*frac)
    y0, x0 = (h - ch)//2, (w - cw)//2
    x = x[y0:y0+ch, x0:x0+cw]

    # Resize
    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    x_t = torch.nn.functional.interpolate(
        x_t, size=(out_size, out_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    # Convert to 3-channel
    x3 = x_t.repeat(3, 1, 1)

    # CRITICAL FIX: Apply normalization
    # This was missing! Pretrained ResNet18 requires ImageNet normalization
    normalize = get_normalization_transform()
    x3 = normalize(x3)

    if return_1ch:
        return x3, x_t[0].cpu().numpy()
    return x3


def load_dicom_for_display(path, dicom_to_hu_func, win01_func,
                           out_size=INPUT_SIZE, frac=FRAC_CROP):
    """
    Load DICOM with same preprocessing as model input for aligned display.
    
    Used for Grad-CAM visualization to ensure spatial alignment.
    
    Args:
        path: Path to DICOM file
        dicom_to_hu_func: Function to convert DICOM to HU
        win01_func: Function to apply windowing
        out_size: Output size (must match model input)
        frac: Center crop fraction (must match model preprocessing)
    
    Returns:
        img01: Preprocessed image (H, W) in [0,1]
    """
    ds = pydicom.dcmread(path, force=True)
    hu = dicom_to_hu_func(ds)
    img01 = win01_func(hu)
    img01_cropped = apply_center_crop(img01, frac=frac)
    img01_resized = cv2.resize(img01_cropped, (out_size, out_size),
                               interpolation=cv2.INTER_LINEAR)
    return img01_resized


def load_dicom_3ch_tensor(path, out_size=224):
    """
    Load DICOM as 3-channel tensor for embedding extraction.
    
    Used in clustering analysis. Does NOT apply ImageNet normalization
    as the pretrained model in clustering uses it internally.
    
    Args:
        path: Path to DICOM file
        out_size: Output size (default 224)
    
    Returns:
        x3: (3, H, W) tensor in [0,1] without normalization
    """
    ds = pydicom.dcmread(path, force=True)
    hu = dicom_to_hu(ds)
    x = win01(hu)

    # Center crop
    h, w = x.shape
    frac = 0.90
    ch, cw = int(h*frac), int(w*frac)
    y0, x0 = (h - ch)//2, (w - cw)//2
    x = x[y0:y0+ch, x0:x0+cw]

    # Resize
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    x_t = F.interpolate(x_t, size=(out_size, out_size),
                       mode="bilinear", align_corners=False).squeeze(0)
    x3 = x_t.repeat(3, 1, 1)
    return x3
