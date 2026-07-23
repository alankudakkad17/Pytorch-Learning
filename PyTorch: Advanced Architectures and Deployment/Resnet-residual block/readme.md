## ResNet Residual Block Lab

This module demonstrates how **residual (skip) connections** improve deep network training stability and performance compared to a plain sequential CNN.

The project builds and compares:

- **PlainCNN** (baseline, no skip connections)
- **SimpleResNet** (ResNet-style model with residual blocks)

Both models are trained on the **Skyview Multi-Landscape Aerial Imagery Dataset** for multi-class landscape classification.

---

## Learning Goals

In this lab you will:

- Build a standard convolutional baseline model (`PlainCNN`)
- Implement a residual architecture (`SimpleResNet`)
- Understand how a `ResidualBlock` works internally:
  - identity shortcut path
  - optional downsampling for shape alignment
  - merge via element-wise addition
- Train and evaluate both models under the same settings
- Compare validation loss/accuracy and confusion matrices

---

## Dataset

- **Name:** Skyview Multi-Landscape Aerial Imagery Dataset  
- **Source:** Kaggle  
- **Classes:** 15 landscape classes  
- **Total Images:** 12,000  
- **Split:** 80% train (9,600) / 20% validation (2,400)

### Class Labels

Agriculture, Airport, Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, Residential, River

---

## Data Pipeline

### Normalization (pre-computed)

- Mean: `[0.378, 0.393, 0.345]`
- Std: `[0.205, 0.173, 0.170]`

### Train transforms

- RandomResizedCrop(100×100, scale=0.8–1.0)
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation(15)
- ColorJitter
- ToTensor + Normalize

### Validation transforms

- Resize(100×100)
- ToTensor + Normalize

---

## Model Architectures

### 1) PlainCNN (Baseline)

- Initial conv stem: `3 -> 32`
- 3 stages of `PlainBlock`s:
  - Stage 1: 32 channels, stride 1
  - Stage 2: 64 channels, stride 2
  - Stage 3: 128 channels, stride 2
- Head:
  - AdaptiveAvgPool2d(1,1)
  - Flatten
  - Linear(128 -> num_classes)

`PlainBlock` = Conv-BN-ReLU -> Conv-BN

---

### 2) SimpleResNet

Same macro-structure as PlainCNN, but each stage uses `ResidualBlock`:

- Main path: Conv-BN-ReLU -> Conv-BN
- Shortcut path:
  - identity when dimensions match
  - 1x1 Conv + BN downsample when stride/channels change
- Merge: `out += identity`, then ReLU

This design enables easier optimization of deeper networks by preserving gradient flow.

---

## Utilities (`helper_utils.py`)

The helper module includes utilities for:

- Dataset statistics display (`display_dataset_stats`)
- Train/validation split creation (`create_datasets`)
- DataLoader setup (`create_dataloaders`)
- Image visualization (`show_sample_images`)
- Torch model summary formatting (`display_torch_summary`)
- Mixed precision training loop (`training_loop_16_mixed`)
- Model comparison plots (`plot_training_logs`)
- Prediction visualization (`visualize_predictions`)
- Confusion matrix + per-class accuracy (`plot_confusion_matrix`)

---

## Training Notes

- Device auto-detection: CUDA / MPS / CPU
- Mixed precision (`autocast`, `GradScaler`) where supported
- Tracks:
  - train loss
  - validation loss
  - macro validation accuracy
  - final confusion matrix

---

## Suggested Experiment Flow

1. Prepare transforms and datasets
2. Build loaders (`batch_size=32`)
3. Initialize both models with same random seed
4. Inspect architecture with `torchinfo.summary`
5. Train both models with identical optimizer/loss/epochs
6. Compare learning curves and final validation accuracy
7. Analyze confusion matrix and class-wise behavior

---

## Files

- `C3_M1_Lab_2_resnet.ipynb` — main lab notebook
- `helper_utils.py` — reusable training/evaluation/visualization utilities
- `readme.md` — module overview (this file)

---

## Requirements

Typical dependencies used in this lab:

- `torch`
- `torchvision`
- `torchmetrics`
- `torchinfo`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `ipython`

---

## Outcome

By completing this lab, you’ll gain practical intuition for why residual learning works and how skip connections can outperform plain deep CNN stacks in real image classification tasks.
