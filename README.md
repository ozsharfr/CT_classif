# CT Brain Classification with Deep Learning

Binary classification of CT brain scans using ResNet18 transfer learning with clustering, early stopping, and Grad-CAM explainability.

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `config.py`:
```python
DICOM_DIR = "path/to/CTs/"
LABELS_CSV = "path/to/labels.csv"
```

### 3. Run Pipeline

**Step 1 - EDA + Clustering Analysis:**
```bash
python EDA.py
```
Output: `outputs/eda_agglomerative.csv`, cluster reports, histograms (~5-10 min)

**Step 2 - Train Classifier:**
```bash
python run_classification.py
```
Output: Trained models, metrics, Grad-CAM visualizations (~10-20 min)

---

## 📁 Code Structure

```
├── config.py                   # ⚙️ Configuration (edit this)
├── utils.py                    # DICOM & image utilities
├── preprocessing.py            # Image preprocessing & augmentation
├── clustering.py               # Embedding extraction & clustering
├── visualization.py            # Plotting & reports
├── model.py                    # Training, early stopping, checkpointing
├── metrics.py                  # Evaluation metrics
├── explainability.py           # Grad-CAM
├── EDA.py                      # Step 1: Run clustering
└── run_classification.py       # Step 2: Run training
```

---

## 📊 Output Structure

```
outputs/
├── eda_agglomerative.csv           # Clustered dataset (from EDA)
├── cluster_report.pdf              # Visual cluster analysis
├── head_frac_hist.pdf              # Quality check histogram
├── models/
│   ├── best_model.pth              # Best validation model
│   ├── final_model.pth             # Final model
│   └── training_history.json       # Training curves
├── test_metrics_by_cluster.csv     # Performance per cluster
└── gradcam/
    ├── gradcam_cluster_18.png      # Explainability visualizations
    ├── gradcam_cluster_37.png
    └── ...
```

---

## ⚙️ Key Configuration

```python
# Training
USE_AUGMENTATION = True          # Toggle augmentation on/off
MAX_EPOCHS = 50                  # Max epochs (early stopping decides)
EARLY_STOPPING_PATIENCE = 5      # Stop after N epochs no improvement
BATCH_SIZE = 32                  # Reduce if GPU memory error

# Model
USE_PRETRAINED = True            # Use ImageNet pretrained weights
MODEL_NAME = 'resnet18'          # Model architecture

# Data Split (60% train, 20% val, 20% test)
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
```

---

## 📈 Expected Output

```
Epoch 1/50: train_loss=0.4523 | val auc=0.887 | [Best]
Epoch 2/50: train_loss=0.3214 | val auc=0.921 | [Best]
...
Epoch 12/50: train_loss=0.0956 | val auc=0.961 | [Best]
Epoch 17/50: Early stopping triggered!

Test Results: acc=0.913 auc=0.968 sens=0.905 spec=0.921
Model saved to: outputs/models/best_model.pth
```

---

## 🎯 Key Features

- ✅ **Transfer Learning** - ResNet18 pretrained on ImageNet
- ✅ **Auto Clustering** - Groups similar images for analysis
- ✅ **Early Stopping** - Stops when validation stops improving
- ✅ **Model Saving** - Auto-saves best model
- ✅ **Augmentation** - Geometric only (optional, toggle on/off)
- ✅ **Class Balancing** - Handles imbalanced datasets
- ✅ **Grad-CAM** - Visual explanations of predictions
- ✅ **Per-Cluster Metrics** - Identifies problematic image groups

---

## 🔧 Common Adjustments

**Disable augmentation:**
```python
USE_AUGMENTATION = False
```

**GPU memory error:**
```python
BATCH_SIZE = 16
```

**More/fewer clusters:**
```python
CLUST_THRESH = 10  # Fewer clusters
CLUST_THRESH = 3   # More clusters
```

---

## 📦 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- pydicom >= 2.3.0
- scikit-learn >= 1.0.0
- pytorch-grad-cam >= 1.4.0

See `requirements.txt` for complete list.

---

## 📝 Labels Format

`labels.csv`:
```csv
ID,Label
patient001,0
patient002,1
```