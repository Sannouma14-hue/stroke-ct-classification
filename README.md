# stroke-ct-classification
Lightweight explainable deep learning for 3-class brain CT stroke classification
# Brain Stroke CT Classification


## Overview

This notebook trains and evaluates three lightweight CNN architectures on a 3-class brain stroke CT dataset. It implements:

- **Two-phase transfer learning** frozen backbone head training → selective fine-tuning of top 20 layers
- **Focal Loss** (γ=2, α=0.25) suppresses easy majority-class examples
- **Balanced oversampling generator**  minority classes oversampled to majority count each epoch
- **Stratified 70/15/15 split**  clean held-out test set with no val/test overlap
- **Optional DICOM denoising**  Gaussian filter (k=5×5, σ=1) for DICOM-exported images only
- **Grad-CAM++**  thresholded spatial explainability heatmaps for each model
- **Efficiency benchmarking**  FLOPs, latency, parameter count vs. reference architectures (ResNet-18, VGG-16, ViT-B/16)

 In this version, the backbone output goes directly to GlobalAveragePooling2D.

---

## Dataset Requirements

**Dataset:** [Brain Stroke CT Dataset](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset) by Ozgur Aslank on Kaggle.

### Expected structure on disk

```
Brain_Stroke_CT_Dataset/
├── Bleeding/
│   └── PNG/
│       ├── image_001.png
│       └── ...
├── Ischemia/
│   └── PNG/
│       ├── image_001.png
│       └── ...
└── Normal/
    └── PNG/
        ├── image_001.png
        └── ...
```

### Image specifications

| Property | Value |
|---|---|
| Format | PNG |
| Input size (resized) | 224 × 224 × 3 |
| Classes | 3 (Bleeding, Ischemia, Normal) |
| Split | 70% train / 15% val / 15% test (stratified) |

> The notebook creates a flat symlink directory (`dataset_flat/`) at runtime — no files are copied or duplicated.

---

## Directory Structure

The notebook uses Kaggle paths by default. Update `BASE_DIR` and `SAVE_DIR` in **Cell 2** if running locally.

```
/kaggle/input/datasets/ozguraslank/brain-stroke-ct-dataset/Brain_Stroke_CT_Dataset/  ← dataset input
/kaggle/working/dataset_flat/       ← auto-created flat symlink directory
/kaggle/working/                    ← all outputs (models, CSVs, plots)
```

**To run locally**, change Cell 2 to:

```python
BASE_DIR = "/your/local/path/to/Brain_Stroke_CT_Dataset"
FLAT_DIR = "/your/local/path/to/dataset_flat"
SAVE_DIR = "/your/local/path/to/outputs"
```

---

## Dependencies

### Core

| Package | Recommended Version | Purpose |
|---|---|---|
| `tensorflow` | ≥ 2.10, < 2.16 | Model training, Keras API |
| `numpy` | ≥ 1.23 | Array operations |
| `pandas` | ≥ 1.5 | Results tables, CSV export |
| `opencv-python` (`cv2`) | ≥ 4.7 | Image loading, augmentation, denoising, CAM overlay |
| `scikit-learn` | ≥ 1.2 | Metrics, stratified splits, class weights |
| `matplotlib` | ≥ 3.6 | Training curves, ROC curves, bar charts |
| `seaborn` | ≥ 0.12 | Confusion matrix heatmaps |

### Built-in 

`os`, `random`, `warnings`, `itertools`, `time`

### Keras applications used

Included with TensorFlow/Keras — no separate install:

- `MobileNetV2`, `MobileNetV3Small`, `EfficientNetB0` (ImageNet weights)

---

## Installation

### Option A Kaggle (recommended)

1. Add the dataset: **Add Data → ozguraslank/brain-stroke-ct-dataset**
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Run all cells in order.

### Option B Local environment

```bash
python -m venv stroke_env
source stroke_env/bin/activate        # Windows: stroke_env\Scripts\activate

pip install tensorflow>=2.10 \
            numpy pandas \
            opencv-python \
            scikit-learn \
            matplotlib seaborn \
            jupyter
```

> For GPU support install `tensorflow[and-cuda]` (TF ≥ 2.13) or configure CUDA + cuDNN manually.

---

## How to Run

Run cells **in order from top to bottom**. Each cell depends on state set by prior cells.

```
Cell 1    → Imports & reproducibility seed
Cell 2    → Configuration (paths, hyperparameters)
Cell 2b   → Build flat symlink directory
Cell 2c   → Stratified 70/15/15 split
Cell 2d   → Compute class weights (train split only)
Cell 2e   → Define BalancedGenerator + NumpyGenerator
Cell 2f   → Define Focal Loss
Cell 3    → Dataset exploration + class distribution chart
Cell 4    → build_generators() function
Cell 5    → build_model() + compile_model() + callbacks
Cell 6    → Training loop (all 3 models, 2 phases each)
Cell 7    → Training curves per model
Cell 8    → Evaluation functions (metrics, confusion matrix)
Cell 9    → Evaluate all models on test set + ROC curves
Cell 10   → Results comparison table + bar charts
Cell 10b  → Efficiency benchmarking (FLOPs, latency, size)
Cell 11   → Grad-CAM++ implementation
Cell 12   → Grad-CAM++ visualisation (best model)
Cell 12b  → Grad-CAM++ visualisation (EfficientNetB0)
Cell 13   → Final summary + list of all saved outputs
```

### Training hyperparameters 

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | (224, 224) | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_FROZEN` | 10 | Phase 1 — frozen backbone |
| `EPOCHS_FINETUNE` | 20 | Phase 2 — fine-tune top layers |
| `LR_HEAD` | 1e-3 | Learning rate, Phase 1 |
| `LR_FINETUNE` | 1e-5 | Learning rate, Phase 2 |
| `UNFREEZE_TOP` | 20 | Number of top backbone layers unfrozen in Phase 2 |
| `SEED` | 42 | Global reproducibility seed |

---

## Outputs

All outputs are saved to `SAVE_DIR` (`/kaggle/working/` by default).

| File | Description |
|---|---|
| `split_train.csv` / `split_val.csv` / `split_test.csv` | Reproducible file-path splits with labels |
| `class_distribution.png` | Class balance bar chart |
| `denoising_sanity_check.png` | Original vs denoised sample (Cell 2b_2 only) |
| `{ModelName}_phase1_best.keras` | Best Phase-1 checkpoint (val AUC) |
| `{ModelName}_phase2_best.keras` | Best Phase-2 checkpoint (val AUC) |
| `{ModelName}_curves.png` | Accuracy / Loss / AUC training curves |
| `{ModelName}_confusion.png` | Counts + normalised confusion matrix |
| `{ModelName}_roc.png` | One-vs-Rest ROC curves with per-class AUC |
| `{ModelName}_gradcam_correct.png` | Grad-CAM++ on correctly classified samples |
| `{ModelName}_gradcam_wrong.png` | Grad-CAM++ on misclassified samples |
| `results_table.csv` | Main performance metrics table |
| `per_class_auc.csv` | Per-class AUC (OvR) for all models |
| `efficiency_table.csv` | Params, size, FLOPs, latency, GPU-hours |
| `results_efficiency_merged.csv` | Performance + efficiency in one table |
| `results_comparison.png` | Grouped bar chart: Accuracy/Sensitivity/Specificity/F1 |
| `per_class_auc_chart.png` | Per-class AUC grouped bar chart |
| `efficiency_params_comparison.png` | Our models vs ResNet/VGG/ViT parameter count |
| `efficiency_flops_comparison.png` | GFLOPs comparison chart |
| `efficiency_latency_comparison.png` | Inference latency comparison chart |

---

## Key Design Decisions
- **Softmax output** 3-class probability distribution; predicted class selected via `argmax`.
- **Two-phase fine-tuning** Phase 1 trains the classification head only (frozen backbone); Phase 2 unfreezes the top 20 backbone layers at LR=1e-5.
- **No val/test leakage** `sklearn.train_test_split` with `stratify=labels` replaces `ImageDataGenerator(validation_split=0.30)`, which was reading val and test from the same 30% slice.
- **Class weights from train split only** prevents leaking val/test class distribution into the loss function.
- **Focal Loss γ=2** down-weights easy majority predictions, focusing training on subtle ischemia and small bleeds.
- **Monitoring `val_auc`** more robust than `val_accuracy` on imbalanced 3-class data; used for both `ModelCheckpoint` and `EarlyStopping`.
- **Grad-CAM++ threshold = 0.40** applied after min-max normalisation to suppress diffuse background activations before overlay.
