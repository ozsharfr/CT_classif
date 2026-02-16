"""
Visualization Module

Functions for generating plots, histograms, and PDF reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_head_frac_hist(df, output_dir, bins=40):
    """
    Save histogram of head fraction distribution.
    
    Args:
        df: DataFrame with 'head_frac' column
        output_dir: Output directory
        bins: Number of histogram bins
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "head_frac_hist.pdf")
    x = df["head_frac"].dropna()

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(7, 5))
        plt.hist(x, bins=bins)
        for q in [0.01, 0.05, 0.5, 0.95, 0.99]:
            val = np.quantile(x, q)
            plt.axvline(val, linestyle="--")
            plt.text(val, plt.ylim()[1]*0.9, f"{int(q*100)}%", rotation=90)
        plt.title("head_frac distribution")
        plt.xlabel("head_frac")
        plt.ylabel("Count")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return out_path


def save_df_hist_pdf(df, output_dir, bins=50):
    """
    Save histograms of all numeric features.
    
    Args:
        df: DataFrame with numeric columns
        output_dir: Output directory
        bins: Number of histogram bins
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "df_histograms.pdf")
    numeric_df = df.select_dtypes(include="number")

    with PdfPages(out_path) as pdf:
        numeric_df.hist(bins=bins, figsize=(12, 10))
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close()
    return out_path


def generate_hu_complete_hist(all_hu, output_dir):
    """
    Generate and save complete HU histogram.
    
    Args:
        all_hu: Dictionary of HU values and their counts
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hu_histogram.png")
    vv = 10 * pd.Series(all_hu) / np.min(list(all_hu.values()))
    pd.Series([[i] * int(v) for i, v in vv.items()]).explode().hist(bins=50)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_cluster_report_pdf(df, Xz, hu_img_func, win01_func,
                           output_dir, min_cluster_size=3):
    """
    Generate PDF report with cluster representatives.
    
    Creates a visual report showing representative images from each cluster,
    separated by label when both labels are present.
    
    Args:
        df: DataFrame with cluster assignments and labels
        Xz: Standardized embeddings array
        hu_img_func: Function to load DICOM as HU
        win01_func: Function to apply windowing
        output_dir: Output directory
        min_cluster_size: Minimum cluster size to include
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cluster_report.pdf")
    cluster_sizes = df["cluster"].value_counts()

    with PdfPages(output_path) as pdf:
        for c, val in cluster_sizes.items():
            if val < min_cluster_size:
                continue

            cluster_df = df[df["cluster"] == c]
            idx_all = cluster_df.index.to_numpy()
            if len(idx_all) == 0:
                continue

            labels_present = cluster_df["label"].unique()
            figs_needed = 2 if len(labels_present) == 2 else 1
            fig, axes = plt.subplots(1, figs_needed, figsize=(6 * figs_needed, 6))
            if figs_needed == 1:
                axes = [axes]

            for i, lbl in enumerate(sorted(labels_present)):
                sub_df = cluster_df[cluster_df["label"] == lbl]
                idx = sub_df.index.to_numpy()
                Xc = Xz[idx]
                center = Xc.mean(axis=0)
                rep_idx = idx[np.argmin(np.linalg.norm(Xc - center, axis=1))]
                row = df.loc[rep_idx]
                _, hu = hu_img_func(row["path"])
                img = win01_func(hu)

                axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
                axes[i].set_title(
                    f"Cluster {c}\nLabel={lbl} | Size={val} | "
                    f"head_frac={row['head_frac']:.2f} | std={row['std']:.2f}"
                )
                axes[i].axis("off")

            if len(labels_present) == 1:
                fig.suptitle(f"Cluster {c} — ONLY label {labels_present[0]}", fontsize=14)
            else:
                fig.suptitle(f"Cluster {c} — label 0 vs label 1", fontsize=14)

            pdf.savefig(fig)
            plt.close(fig)

    return output_path
