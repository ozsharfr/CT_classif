"""
CT Brain Classification with Transfer Learning

This script trains a binary classifier for CT brain images using transfer
learning with ResNet18. It includes class balancing, per-cluster evaluation,
and Grad-CAM explainability visualization.

Usage:
    python run_classification.py

Outputs:
    - Console: Training metrics per epoch
    - outputs/test_metrics_by_cluster.csv: Per-cluster performance
    - outputs/gradcam/*.png: Grad-CAM visualizations per cluster
"""

import os
import json
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split

from config import (
    EDA_CSV, BATCH_SIZE, MAX_EPOCHS, LR, FREEZE_LAYERS, SEED, DEVICE,
    VALIDATION_SIZE, TEST_SIZE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_METRIC, EARLY_STOPPING_MIN_DELTA,
    MODEL_CHECKPOINT_DIR, SAVE_BEST_MODEL, SAVE_FINAL_MODEL, SAVE_TRAINING_HISTORY,
    MODEL_NAME, USE_PRETRAINED,
    USE_AUGMENTATION, COMPUTE_NORMALIZATION_FROM_DATASET
)
from utils import dicom_to_hu, win01
from preprocessing import load_dicom_as_tensor, compute_dataset_statistics
from model import CTDataset, train_one_epoch, predict, EarlyStopping, save_checkpoint, load_checkpoint
from metrics import metrics_binary, report_by_cluster
from explainability import save_explainability_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def prepare_data(df):
    """
    Prepare and split data for training.
    
    Creates 3-way split: train/val/test (60%/20%/20% by default)
    
    Args:
        df: DataFrame from EDA with path, label, cluster columns
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Filter low-quality scans
    df = df[df['head_frac'] > 0.05]
    df = df[df["path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    df["label"] = df["label"].astype(int)

    logger.info("DATA SUMMARY")
    logger.info("-"*60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label prevalence: {df['label'].mean():.3f}")
    logger.info(f"Number of clusters: {df['cluster'].nunique()}")

    # Stratified 3-way split by patient
    ids = df["PatientID"].values
    y = df["label"].values
    unique_ids = pd.Series(ids).unique()
    
    # Get one label per patient for stratification
    patient_labels = pd.Series(y, index=ids).groupby(level=0).max().loc[unique_ids].values
    
    # Step 1: Split into train+val (80%) and test (20%)
    train_val_ids, test_ids = train_test_split(
        unique_ids, test_size=TEST_SIZE, random_state=SEED,
        stratify=patient_labels
    )
    
    # Step 2: Split train+val into train (75% of 80% = 60%) and val (25% of 80% = 20%)
    train_val_labels = pd.Series(y, index=ids).groupby(level=0).max().loc[train_val_ids].values
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=SEED,
        stratify=train_val_labels
    )

    train_df = df[df["PatientID"].isin(train_ids)].copy()
    val_df = df[df["PatientID"].isin(val_ids)].copy()
    test_df = df[df["PatientID"].isin(test_ids)].copy()
    
    logger.info(f"Train: {len(train_df)} ({len(train_ids)} patients)")
    logger.info(f"Val: {len(val_df)} ({len(val_ids)} patients)")
    logger.info(f"Test: {len(test_df)} ({len(test_ids)} patients)")
    
    return train_df, val_df, test_df


def build_model(train_df):
    """
    Build ResNet18 model with class-balanced loss.
    
    Args:
        train_df: Training DataFrame
    
    Returns:
        tuple: (model, loss_fn, optimizer)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers except the final fully connected layer
    if FREEZE_LAYERS:
        layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
    
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    # Class balancing
    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-9)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    logger.info(f"Class balance: pos={pos}, neg={neg}, weight={pos_weight.item():.3f}")
    
    return model, loss_fn, optim


def train_model(model, train_loader, val_loader, loss_fn, optim):
    """
    Train model with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optim: Optimizer
    
    Returns:
        Dictionary with training history
    """
    logger.info("="*60)
    logger.info("TRAINING")
    logger.info("="*60)
    
    # Initialize early stopping
    mode = 'max' if EARLY_STOPPING_METRIC in ['auc', 'acc'] else 'min'
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode=mode
    )
    
    best_val_score = -float('inf') if mode == 'max' else float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn)
        
        # Validate
        y_prob, y_true, clusters = predict(model, val_loader)
        acc, auc, recall, precision, cm = metrics_binary(y_true, y_prob, thr=0.5)
        
        val_metrics = {
            'acc': acc,
            'auc': auc,
            'recall': recall,
            'precision': precision,
            'loss': tr_loss
        }
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}/{MAX_EPOCHS}: train_loss={tr_loss:.4f} | "
            f"val acc={acc:.3f} auc={auc:.3f} recall={recall:.3f} precision={precision:.3f}"
        )
        
        # Save history
        history['train_loss'].append(tr_loss)
        history['val_metrics'].append(val_metrics)
        
        # Get current metric for early stopping
        current_score = val_metrics[EARLY_STOPPING_METRIC]
        
        # Save best model
        is_best = False
        if mode == 'max':
            is_best = current_score > best_val_score
        else:
            is_best = current_score < best_val_score
        
        if is_best and SAVE_BEST_MODEL:
            best_val_score = current_score
            best_epoch = epoch
            save_checkpoint(
                model, optim, epoch, val_metrics,
                os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
            )
        
        # Check early stopping
        if early_stopping(current_score, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best {EARLY_STOPPING_METRIC}: {best_val_score:.4f} at epoch {best_epoch}")
            break
    
    # Load best model
    if SAVE_BEST_MODEL and os.path.exists(os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")):
        logger.info(f"Loading best model from epoch {best_epoch}")
        load_checkpoint(os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth"), model, optim)
    
    return history


def evaluate_and_save(model, test_loader, test_df, history):
    """
    Evaluate model on test set and save results.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        test_df: Test DataFrame
        history: Training history dictionary
    """
    logger.info("="*60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*60)
    
    # Overall metrics
    y_prob, y_true, clusters = predict(model, test_loader)
    acc, auc, recall, precision, cm = metrics_binary(y_true, y_prob, thr=0.5)
    
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy:    {acc:.3f}")
    logger.info(f"  AUC:         {auc:.3f}")
    logger.info(f"  recall: {recall:.3f}")
    logger.info(f"  precision: {precision:.3f}")
    logger.info(f"  Confusion Matrix: {cm}")
    
    # Per-cluster metrics
    logger.info("-"*60)
    logger.info("PER-CLUSTER METRICS")
    logger.info("-"*60)
    
    rep = report_by_cluster(y_true, y_prob, clusters, min_cluster_size=3, thr=0.5)
    out_path = os.path.join(os.path.dirname(EDA_CSV), "test_metrics_by_cluster.csv")
    rep.to_csv(out_path, index=False)
    logger.info(f"\n{rep.head(20)}")
    logger.info(f"Saved to: {out_path}")
    
    # Save training history
    if SAVE_TRAINING_HISTORY:
        history_path = os.path.join(MODEL_CHECKPOINT_DIR, "training_history.json")
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")


def main():
    logger.info("="*60)
    logger.info("CT BRAIN CLASSIFICATION - TRAINING")
    logger.info("="*60)
    
    # Log configuration
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Pretrained: {USE_PRETRAINED}")
    logger.info(f"Augmentation: {USE_AUGMENTATION}")
    logger.info(f"Max Epochs: {MAX_EPOCHS}")
    logger.info(f"Early Stopping: {EARLY_STOPPING_PATIENCE} patience on {EARLY_STOPPING_METRIC}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    
    # Load and prepare data
    df = pd.read_csv(EDA_CSV)
    train_df, val_df, test_df = prepare_data(df)
    
    # Compute dataset statistics if training from scratch
    if not USE_PRETRAINED and COMPUTE_NORMALIZATION_FROM_DATASET:
        logger.info("Computing dataset normalization statistics...")
        
        # Create temporary loader without augmentation
        temp_dataset = CTDataset(train_df, training=False)
        temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Compute and update config
        dataset_mean, dataset_std = compute_dataset_statistics(temp_loader)
        import config
        config.DATASET_MEAN = dataset_mean
        config.DATASET_STD = dataset_std
        
        logger.info(f"Dataset normalization: mean={dataset_mean}, std={dataset_std}")

    # Create data loaders
    train_loader = DataLoader(
        CTDataset(train_df, training=True),  # Augmentation ON for training
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        CTDataset(val_df, training=False),  # Augmentation OFF for validation
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        CTDataset(test_df, training=False),  # Augmentation OFF for test
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Build model
    model, loss_fn, optim = build_model(train_df)

    # Train with early stopping
    history = train_model(model, train_loader, val_loader, loss_fn, optim)

    # Save final model
    if SAVE_FINAL_MODEL:
        save_checkpoint(
            model, optim, MAX_EPOCHS, {},
            os.path.join(MODEL_CHECKPOINT_DIR, "final_model.pth")
        )

    # Evaluate on test set
    evaluate_and_save(model, test_loader, test_df, history)

    # Grad-CAM explainability
    logger.info("="*60)
    logger.info("GENERATING GRAD-CAM VISUALIZATIONS")
    logger.info("="*60)
    
    gradcam_dir = os.path.join(os.path.dirname(EDA_CSV), "gradcam")
    save_explainability_report(
        df_test=test_df,
        model=model,
        load_tensor_func=load_dicom_as_tensor,
        win01_func=win01,
        dicom_to_hu_func=dicom_to_hu,
        device=DEVICE,
        out_dir=gradcam_dir
    )
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best model saved to: {os.path.join(MODEL_CHECKPOINT_DIR, 'best_model.pth')}")
    if SAVE_TRAINING_HISTORY:
        logger.info(f"Training history: {os.path.join(MODEL_CHECKPOINT_DIR, 'training_history.json')}")


if __name__ == "__main__":
    main()
