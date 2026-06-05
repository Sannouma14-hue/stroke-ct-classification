# Brain Stroke CT Classification
### Lightweight Explainable Deep Learning — 3-Class CT Triage

**Author:** Tasnim Mohamed Anzize  
**Dataset:** [TEKNOFEST-2021 Brain Stroke CT Dataset](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset)  
**Task:** Three-class classification — Normal / Ischemia / Bleeding (Hemorrhagic)

---

## What this notebook does

This notebook trains and evaluates three compact transfer-learning architectures for automated stroke classification from brain CT images. The goal is to build a system small enough to run without GPU infrastructure, while remaining clinically meaningful as a triage aid in low-resource settings.

Three backbones are compared under identical conditions:

| Model | Parameters | GFLOPs | Size |
|---|---|---|---|
| MobileNetV3Small | 1.27M | 0.11 | 14.8 MB |
| MobileNetV2 | 2.95M | 0.59 | 34.6 MB |
| EfficientNetB0 | 4.75M | 0.73 | 51.3 MB |

**Best model: MobileNetV2** — 92.59% accuracy, F1 89.56%, AUC 0.9816

---

## Dataset

The TEKNOFEST-2021 Brain Stroke CT Dataset contains 6,650 axial CT brain scans collected from Turkish emergency departments in 2019–2020, annotated by seven radiologists across three classes:

| Class | Count | Percentage |
|---|---|---|
| Normal | 4,427 | 66.6% |
| Ischemia | 1,130 | 17.0% |
| Bleeding | 1,093 | 16.4% |

The dataset reflects real-world emergency radiology distribution, with normal cases as the majority class.

---

## Pipeline overview

### 1. Data preparation
- Flat directory structure built via symlinks (no data copy)
- Stratified 70/15/15 split using `sklearn.train_test_split` with `stratify=labels` and `seed=42`
- Clean held-out test set that shares no overlap with validation

### 2. Class imbalance handling — three strategies combined
- **BalancedGenerator** — oversamples minority classes to match majority count each epoch
- **Class weights** — computed from training labels only (`sklearn` balanced weighting)
- **Focal Loss** — gamma=2.0, alpha=0.25; down-weights easy majority examples

### 3. Architecture
Each backbone uses a shared classification head:
```
Backbone (ImageNet pretrained, frozen in Phase 1)
    → Spatial Attention Block
        → att_conv1 (256 filters, 1x1, ReLU)
        → att_bn (BatchNorm)
        → att_map (1 filter, 1x1, Sigmoid)   ← Grad-CAM++ target layer
        → attended_features (Multiply)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(256, ReLU) → Dropout(0.4)
    → Dense(128, ReLU) → Dropout(0.3)
    → Dense(3, Softmax)
```

The spatial attention block forces the model to localize lesions before feature aggregation, and serves as the unified Grad-CAM++ target layer across all three models.

### 4. Two-phase training
| Phase | Epochs | LR | Backbone |
|---|---|---|---|
| Phase 1 — head only | 15 max | 1e-3 | Frozen |
| Phase 2 — fine-tuning | 50 max | 1e-5 | Top 80 layers unfrozen |

Callbacks: `EarlyStopping` (val_auc, patience=5), `ReduceLROnPlateau` (factor=0.5, patience=3), `ModelCheckpoint` (best val_auc saved).

### 5. Evaluation
All models evaluated on the held-out test set (998 images) using:
- Accuracy, Sensitivity, Specificity, Precision, macro F1-score
- Macro AUC and per-class AUC (one-vs-rest)
- Efficiency metrics: parameters, GFLOPs, model size, inference latency

### 6. Explainability
Grad-CAM++ applied to the best-performing model using `att_map` as the target layer. Heatmaps overlaid on original CT images with JET colormap at 0.45 blending and a 0.40 activation threshold to suppress background noise.

---

## Results

### Classification performance (test set, 998 images)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC |
|---|---|---|---|---|---|---|
| **MobileNetV2** | **92.59%** | **90.91%** | **96.13%** | **88.42%** | **89.56%** | **0.9816** |
| EfficientNetB0 | 90.88% | 89.13% | 94.79% | 87.06% | 87.87% | 0.9776 |
| MobileNetV3Small | 87.98% | 86.79% | 93.81% | 82.31% | 84.16% | 0.9670 |

### Per-class AUC (one-vs-rest)

| Model | AUC Bleeding | AUC Ischemia | AUC Normal |
|---|---|---|---|
| MobileNetV2 | 0.9796 | 0.9767 | 0.9884 |
| EfficientNetB0 | 0.9802 | 0.9770 | 0.9755 |
| MobileNetV3Small | 0.9753 | 0.9592 | 0.9665 |

### MobileNetV2 — per-class breakdown

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bleeding | 0.89 | 0.90 | 0.89 |
| Ischemia | 0.79 | 0.89 | 0.84 |
| Normal | 0.98 | 0.94 | 0.96 |

---

## Output files

| File | Description |
|---|---|
| `results_table.csv` | Global metrics for all three models |
| `per_class_auc.csv` | Per-class AUC scores |
| `efficiency_table.csv` | Parameters, GFLOPs, size, latency |
| `[model]_curves.png` | Training/validation accuracy, loss, AUC curves |
| `[model]_confusion.png` | Confusion matrix on test set |
| `[model]_roc.png` | ROC curves per class |
| `[model]_gradcam_correct.png` | Grad-CAM++ on correctly classified samples |
| `[model]_gradcam_wrong.png` | Grad-CAM++ on misclassified samples |
| `class_distribution.png` | Dataset class balance chart |

---

## Key design decisions

- **Stratified split** — fixes the common `validation_split` error where val and test generators read the same data slice
- **Model-specific preprocessing** — each backbone uses its own `preprocess_input` function, essential for transfer learning correctness
- **val_auc as checkpoint metric** — more robust than accuracy under class imbalance
- **No vertical flip** — clinically conservative augmentation; vertical flip is not a realistic CT acquisition orientation
- **Unified Grad-CAM++ target** — `att_map` is used for all three models, making heatmaps directly comparable across architectures

---

## Clinical motivation

Stroke accounts for nearly 90% of its global mortality burden in low- and middle-income countries, where access to specialist CT interpretation is severely limited. A lightweight model deployable on standard CPU hardware can serve as a preliminary triage aid — rapidly excluding hemorrhagic stroke where tPA is contraindicated, and flagging suspected ischemic cases for urgent specialist review.

---

## Requirements

```
tensorflow >= 2.19.0
scikit-learn
opencv-python
numpy
pandas
matplotlib
seaborn
```

---

## Citation

If you use this notebook, please cite the dataset:

> Koç U. et al., "Artificial Intelligence in Healthcare Competition (TEKNOFEST-2021): Stroke Data Set," *Eurasian Journal of Medicine*, vol. 54, no. 3, pp. 248–258, 2022. DOI: 10.5152/eurasianjmed.2022.22096
