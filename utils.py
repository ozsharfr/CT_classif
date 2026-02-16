"""
Utility Functions Module

General-purpose helper functions for DICOM processing and image manipulation.
These functions are used across multiple modules.
"""

import numpy as np
import pydicom
from config import WMIN, WMAX


def dicom_to_hu(ds):
    """
    Convert DICOM pixel array to Hounsfield Units.
    
    Args:
        ds: pydicom Dataset object
    
    Returns:
        numpy array of HU values
    """
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img * slope + intercept


def win01(hu):
    """
    Apply brain window and normalize to [0,1].
    
    Uses global WMIN, WMAX from config.
    
    Args:
        hu: Hounsfield Unit array
    
    Returns:
        Normalized array in [0,1]
    """
    x = np.clip(hu, WMIN, WMAX)
    return (x - WMIN) / (WMAX - WMIN + 1e-6)


def hu_img(path):
    """
    Read DICOM file and convert to Hounsfield Units.
    
    Args:
        path: Path to DICOM file
    
    Returns:
        tuple: (patient_id, hu_array)
    """
    ds = pydicom.dcmread(path, force=True)
    hu = dicom_to_hu(ds)
    pid = str(getattr(ds, "PatientID", "UNK"))
    return pid, hu


def apply_center_crop(img, frac=0.90):
    """
    Apply center crop to image.
    
    Args:
        img: Input image array (H, W) or (H, W, C)
        frac: Fraction of image to keep (0-1)
    
    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    ch, cw = int(h*frac), int(w*frac)
    y0, x0 = (h - ch)//2, (w - cw)//2
    return img[y0:y0+ch, x0:x0+cw]
