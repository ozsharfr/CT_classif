"""
Model Module

PyTorch dataset class and training/inference functions.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import logging

from config import DEVICE
from preprocessing import load_dicom_as_tensor, get_augmentation_transform

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in metric to qualify as an improvement
        mode: 'max' for metrics like AUC/accuracy, 'min' for loss
    """
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, current_score, epoch):
        """
        Call after each epoch with current metric value.
        
        Returns:
            True if should stop training, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            logger.info(f"Initial best {self.mode} score: {current_score:.4f}")
            return False
        
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            logger.info(f"New best {self.mode} score: {current_score:.4f}")
            return False
        else:
            self.counter += 1
            logger.info(f"No improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                self.early_stop = True
                return True
            
            return False


def save_checkpoint(model, optimizer, epoch, metrics, path, config_dict=None):
    """
    Save model checkpoint with all metadata.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'auc': 0.95, 'acc': 0.89})
        path: Path to save checkpoint
        config_dict: Optional configuration dictionary to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if config_dict:
        checkpoint['config'] = config_dict
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to: {path}")
    logger.info(f"Metrics: {metrics}")


def load_checkpoint(path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Dictionary with epoch and metrics
    """
    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from: {path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }



class CTDataset(Dataset):
    """
    CT brain image dataset for PyTorch.
    
    Loads DICOM images on-the-fly with preprocessing and optional augmentation.
    
    Args:
        df: DataFrame with 'path', 'label', and 'cluster' columns
        training: If True, applies data augmentation (if enabled in config)
    """
    
    def __init__(self, df, training=False):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.augmentation = get_augmentation_transform(training)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        Returns:
            tuple: (image_tensor, label, cluster_id)
        """
        row = self.df.iloc[i]
        x = load_dicom_as_tensor(row["path"])
        
        # Apply augmentation if training (and if enabled in config)
        if self.augmentation is not None:
            x = self.augmentation(x)
        
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y, row["cluster"]


def train_one_epoch(model, loader, optim, loss_fn):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optim: Optimizer
        loss_fn: Loss function
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    losses = []
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict(model, loader):
    """
    Get predictions for entire dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader
    
    Returns:
        tuple: (probabilities, true_labels, cluster_ids)
    """
    model.eval()
    probs, ys, clusters = [], [], []
    for x, y, c in loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
        clusters.append(np.array(c))
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(clusters)
