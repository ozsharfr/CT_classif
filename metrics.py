"""
Metrics Module

Functions for computing and reporting classification performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def metrics_binary(y_true, y_prob, thr=0.5):
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities
        thr: Classification threshold (default 0.5)
    
    Returns:
        tuple: (accuracy, auc, recall, precision, confusion_matrix)
    """
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    auc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    return acc, auc, recall, precision, (tn, fp, fn, tp)


def report_by_cluster(y_true, y_prob, clusters, min_cluster_size=3, thr=0.5):
    """
    Generate per-cluster performance metrics.
    
    Creates a DataFrame with performance metrics for each cluster,
    allowing analysis of model performance across different image types.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        clusters: Cluster assignments
        min_cluster_size: Minimum cluster size to include
        thr: Classification threshold
    
    Returns:
        DataFrame with columns: cluster, n, pos_rate, acc, auc, recall, precision, confusion_matrix
    """
    out = []
    for c in sorted(np.unique(clusters)):
        mask = (clusters == c)
        if mask.sum() < min_cluster_size:
            continue
        acc, auc, recall, precision, cm = metrics_binary(y_true[mask], y_prob[mask], thr=thr)
        out.append({
            "cluster": int(c),
            "n": int(mask.sum()),
            "pos_rate": float(y_true[mask].mean()),
            "acc@0.5": acc,
            "auc": auc,
            "recall@0.5": recall,
            "precision@0.5": precision,
            "TN_FP_FN_TP": cm
        })
    return pd.DataFrame(out).sort_values("n", ascending=False)
