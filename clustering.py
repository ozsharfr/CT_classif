"""
Clustering Module

Functions for extracting embeddings and performing hierarchical clustering.
Uses pretrained ResNet18 for feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scipy.spatial.distance import pdist

from config import DEVICE
from preprocessing import load_dicom_3ch_tensor


def build_embedding_model():
    """
    Build ResNet18 model for feature extraction.
    
    Replaces final FC layer with identity to output 512-d embeddings.
    
    Returns:
        PyTorch model in eval mode
    """
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Identity()  # Output 512-d embeddings
    m.eval().to(DEVICE)
    return m


@torch.no_grad()
def extract_embedding(model, path):
    """
    Extract 512-d embedding from DICOM image.
    
    Args:
        model: ResNet18 embedding model
        path: Path to DICOM file
    
    Returns:
        numpy array of shape (512,)
    """
    x = load_dicom_3ch_tensor(path).unsqueeze(0).to(DEVICE)
    emb = model(x).squeeze(0).detach().cpu().numpy()
    return emb.astype(np.float32)


def choose_threshold_from_pdist(E, q=10, sample_size=2000, metric="euclidean", seed=42):
    """
    Compute adaptive clustering threshold from distance distribution.
    
    Samples a subset of embeddings to compute pairwise distances efficiently,
    then returns the q-th percentile as the threshold.
    
    Args:
        E: Embeddings array (N, D)
        q: Percentile for threshold (lower = more clusters)
        sample_size: Sample size for distance computation
        metric: Distance metric (default: euclidean)
        seed: Random seed
    
    Returns:
        tuple: (threshold, distance_array)
    """
    rng = np.random.default_rng(seed)
    N = E.shape[0]
    m = min(sample_size, N)
    idx = rng.choice(N, size=m, replace=False)
    d = pdist(E[idx], metric=metric)
    thr = float(np.percentile(d, q))
    return thr, d
