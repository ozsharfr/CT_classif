"""
Exploratory Data Analysis (EDA) for CT Brain Images

This script performs clustering analysis on CT brain scans to identify
image groups with similar characteristics.

Usage:
    python EDA.py

Outputs:
    - outputs/eda_agglomerative.csv: Processed dataset with cluster assignments
    - outputs/cluster_report.pdf: Visual report of cluster representatives
    - outputs/*.pdf/*.png: Various histograms and distributions
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from config import (
    DICOM_DIR, OUTPUT_DIR, LABELS_CSV,
    CLUST_THRESH, MIN_CLUST_SIZE, NON_AIR
)
from utils import hu_img, win01
from clustering import build_embedding_model, extract_embedding, choose_threshold_from_pdist
from visualization import (
    save_head_frac_hist, save_df_hist_pdf,
    generate_hu_complete_hist, save_cluster_report_pdf
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_basic_features(paths):
    """
    Extract basic statistical features from DICOM files.
    
    Args:
        paths: List of DICOM file paths
    
    Returns:
        tuple: (DataFrame with features, HU histogram dictionary)
    """
    rows = []
    all_hu = {}
    
    for p in paths:
        pid, hu = hu_img(p)
        mask = (hu > NON_AIR)
        x = win01(hu)
        vals = x[mask] if mask.any() else x.ravel()

        rows.append(dict(
            path=p,
            PatientID=pid,
            head_frac=float(mask.mean()),
            mean=float(vals.mean()),
            std=float(vals.std())
        ))
        vals1 = hu[mask] if mask.any() else hu.ravel()
        for k, v in pd.Series(vals1).value_counts().to_dict().items():
            all_hu[k] = all_hu.get(k, 0) + v

    return pd.DataFrame(rows), all_hu


def perform_clustering(df, embeddings):
    """
    Perform hierarchical clustering on embeddings.
    
    Args:
        df: DataFrame with image features
        embeddings: Standardized embedding array
    
    Returns:
        DataFrame with cluster assignments added
    """
    logger.info("Performing hierarchical clustering...")
    thr, d = choose_threshold_from_pdist(embeddings, q=CLUST_THRESH,
                                        sample_size=2000, metric="euclidean")
    logger.info(f"Clustering threshold (at {CLUST_THRESH}th percentile): {thr:.4f}")
    
    clu = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thr,
        linkage="ward"
    )
    labels = clu.fit_predict(embeddings)
    df["cluster"] = labels
    
    logger.info("Cluster sizes:")
    logger.info(f"\n{df['cluster'].value_counts().sort_index()}")
    
    return df


def main():
    logger.info("="*60)
    logger.info("CT BRAIN IMAGE CLUSTERING - EDA")
    logger.info("="*60)
    
    # Collect DICOM files
    paths = glob.glob(os.path.join(DICOM_DIR, "**", "*.dcm"), recursive=True)
    logger.info(f"Total DICOM files found: {len(paths)}")

    # Extract features
    logger.info("Extracting basic features...")
    df, all_hu = extract_basic_features(paths)

    # Generate visualizations
    logger.info("Generating histograms...")
    generate_hu_complete_hist(all_hu, output_dir=OUTPUT_DIR)
    save_head_frac_hist(df, OUTPUT_DIR, bins=50)
    save_df_hist_pdf(df, OUTPUT_DIR, bins=50)

    # Image statistics
    counts = df.groupby("PatientID").size()
    logger.info("Images per PatientID:")
    logger.info(f"\n{counts.describe()}")
    logger.info(f"Fraction with exactly 1 image: {float((counts == 1).mean()):.3f}")

    # Extract embeddings for clustering
    logger.info("Building embedding model...")
    emb_model = build_embedding_model()
    
    logger.info("Extracting embeddings (this may take a while)...")
    E = np.vstack([extract_embedding(emb_model, p) for p in df["path"]])
    embeddings = StandardScaler().fit_transform(E)

    # Clustering
    df = perform_clustering(df, embeddings)

    # Merge with labels
    logger.info("Merging with labels...")
    lab = pd.read_csv(LABELS_CSV)
    lab = lab.rename(columns={"ID": "PatientID", "Label": "label"})
    df["PatientID"] = df["path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    df = df.merge(lab[["PatientID", "label"]], on="PatientID", how="inner")
    logger.info(f"Total samples after merge: {len(df)}")

    # Generate cluster report
    logger.info("Generating cluster report...")
    save_cluster_report_pdf(
        df=df, Xz=embeddings,
        hu_img_func=hu_img, win01_func=win01,
        output_dir=OUTPUT_DIR, min_cluster_size=MIN_CLUST_SIZE
    )

    # Save results
    out_csv = os.path.join(OUTPUT_DIR, "eda_agglomerative.csv")
    df.to_csv(out_csv, index=False)
    logger.info("="*60)
    logger.info("EDA COMPLETE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {out_csv}")
    logger.info(f"Cluster report: {os.path.join(OUTPUT_DIR, 'cluster_report.pdf')}")


if __name__ == "__main__":
    main()
