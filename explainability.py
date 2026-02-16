"""
Grad-CAM Explainability Utilities

This module provides functions for generating Grad-CAM visualizations
to explain CT brain classification model decisions. The implementation
ensures proper spatial alignment between heatmaps and images.

Key Features:
- Grad-CAM generation with proper normalization handling
- Aligned overlay of heatmaps on original images
- Per-cluster visualization reports
"""

import os
import logging
import numpy as np
import cv2
import torch
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def apply_center_crop(img, frac=0.90):
    """
    Apply center crop to image.
    
    Args:
        img: Input image array
        frac: Fraction of image to keep (0-1)
    
    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    ch, cw = int(h*frac), int(w*frac)
    y0, x0 = (h - ch)//2, (w - cw)//2
    return img[y0:y0+ch, x0:x0+cw]


def gradcam_on_one(model, img_tensor_3chw, target_layer, device):
    """
    Generate Grad-CAM for a single image.
    
    Args:
        model: PyTorch model
        img_tensor_3chw: Input tensor (3, H, W) with ImageNet normalization
        target_layer: Target layer for Grad-CAM
        device: Device (cpu/cuda)
    
    Returns:
        cam: Grad-CAM heatmap (H, W) in [0,1]
    """
    model.eval()
    input_tensor = img_tensor_3chw.unsqueeze(0).to(device)
    cam_extractor = GradCAM(model=model, target_layers=[target_layer])
    
    try:
        targets = [BinaryClassifierOutputTarget(1)]
        grayscale_cam = cam_extractor(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False
        )[0]
    finally:
        if hasattr(cam_extractor, 'activations_and_grads'):
            cam_extractor.activations_and_grads.release()
        del cam_extractor

    return grayscale_cam


def overlay_cam_on_image(img01_hw, cam_hw, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on grayscale image.
    
    Args:
        img01_hw: Grayscale image (H, W) in [0,1]
        cam_hw: CAM heatmap (H, W) in [0,1]
        alpha: Transparency of heatmap overlay
    
    Returns:
        RGB visualization (H, W, 3) as uint8
    """
    img01_hw = np.asarray(img01_hw, dtype=np.float32)
    img01_hw = np.clip(img01_hw, 0, 1)
    H, W = img01_hw.shape[:2]

    cam_hw = np.asarray(cam_hw, dtype=np.float32)
    if cam_hw.shape != (H, W):
        cam_hw = cv2.resize(cam_hw, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_hw = np.clip(cam_hw, 0, 1)

    img_rgb = np.stack([img01_hw, img01_hw, img01_hw], axis=-1)
    visualization = show_cam_on_image(
        img_rgb, cam_hw, use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=1-alpha
    )
    return visualization


def load_dicom_for_display(path, dicom_to_hu_func, win01_func, 
                           out_size=224, frac=0.90):
    """
    Load DICOM with same preprocessing as model input for aligned display.
    
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


def save_explainability_report(
    df_test, model, load_tensor_func, win01_func, dicom_to_hu_func,
    device, out_dir, target_layer_name="layer4",
    max_per_cluster=6, only_clusters_with_n_ge=10,
    out_size=224, crop_frac=0.90
):
    """
    Generate Grad-CAM explainability report for test set.
    
    This function creates visualizations showing which brain regions
    the model focuses on when making predictions. Images are grouped
    by cluster and prediction type (TP/FN/FP/TN).
    
    Args:
        df_test: Test DataFrame with path, label, cluster columns
        model: Trained PyTorch model
        load_tensor_func: Function to load DICOM as tensor
        win01_func: Function for brain windowing
        dicom_to_hu_func: Function to convert DICOM to HU
        device: Device (cpu/cuda)
        out_dir: Output directory for visualizations
        target_layer_name: Model layer for Grad-CAM (default: layer4)
        max_per_cluster: Max images to show per cluster
        only_clusters_with_n_ge: Min cluster size to visualize
        out_size: Model input size (must match training)
        crop_frac: Center crop fraction (must match training)
    
    Outputs:
        PNG files: gradcam_cluster_*.png in out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Generating Grad-CAM explainability report for {len(df_test)} test samples...")

    target_layer = dict(model.named_modules())[target_layer_name]
    sizes = df_test["cluster"].value_counts()
    clusters_sorted = sizes.index.tolist()

    for c in clusters_sorted:
        if sizes[c] < only_clusters_with_n_ge:
            continue

        sub = df_test[df_test["cluster"] == c].copy()

        # Calculate predictions
        probs = []
        ys = sub["label"].values.astype(int)
        for p in sub["path"].values:
            x = load_tensor_func(p).to(device)
            with torch.no_grad():
                logit = model(x.unsqueeze(0)).squeeze().item()
                prob = 1 / (1 + np.exp(-logit))
            probs.append(prob)
        sub["prob"] = probs
        sub["pred"] = (sub["prob"] >= 0.5).astype(int)

        def pick(kind):
            """Pick representative samples by prediction type."""
            if kind == "TP":
                s = sub[(sub.label==1)&(sub.pred==1)].sort_values("prob", ascending=False)
            elif kind == "FN":
                s = sub[(sub.label==1)&(sub.pred==0)].sort_values("prob", ascending=True)
            elif kind == "FP":
                s = sub[(sub.label==0)&(sub.pred==1)].sort_values("prob", ascending=False)
            else:
                s = sub[(sub.label==0)&(sub.pred==0)].sort_values("prob", ascending=True)
            return s.head(max_per_cluster//3 if max_per_cluster>=3 else max_per_cluster)

        picks = pd.concat([pick("TP"), pick("FN"), pick("FP")], 
                         axis=0).drop_duplicates().head(max_per_cluster)
        if len(picks) == 0:
            continue

        # Create visualization grid
        n = len(picks)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(5*cols, 5*rows))
        fig.suptitle(f"Cluster {c} | n={sizes[c]}", fontsize=16, fontweight='bold')

        for i, row in enumerate(picks.itertuples(), start=1):
            path = row.path
            try:
                img01_display = load_dicom_for_display(
                    path, dicom_to_hu_func, win01_func,
                    out_size=out_size, frac=crop_frac
                )
                x3 = load_tensor_func(path)
                cam = gradcam_on_one(model, x3, target_layer, device)
                overlay_rgb = overlay_cam_on_image(img01_display, cam, alpha=0.45)

                ax = fig.add_subplot(rows, cols, i)
                ax.imshow(overlay_rgb)
                
                kind = "TP" if (row.label==1 and row.pred==1) else \
                       "FN" if (row.label==1 and row.pred==0) else \
                       "FP" if (row.label==0 and row.pred==1) else "TN"
                ax.set_title(f"{kind}: GT={row.label}, Pred={row.pred}, P={row.prob:.3f}",
                           fontsize=12, fontweight='bold')
                ax.axis("off")
            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")
                continue

        out_path = os.path.join(out_dir, f"gradcam_cluster_{c}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved cluster {c} ({len(picks)} samples)")

    logger.info(f"Saved Grad-CAM images to: {out_dir}")
    logger.info("Images and CAMs are properly aligned!")
