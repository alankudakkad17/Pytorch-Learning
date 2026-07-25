# Siamese

This folder contains a Jupyter notebook lab and helper utilities for building and training Siamese networks for similarity learning.

## Contents

- `C3_M1_Lab_1_siamese_network.ipynb` — the main lab notebook.
- `helper_utils.py` — dataset preparation, visualization, inference, and evaluation helpers.
- `training_functions.py` — training and evaluation loops used by the notebook.

## What the lab covers

The notebook demonstrates two use cases:

1. **Signature verification**
   - Builds a triplet dataset from `Real/` and `Fake/` signature folders organized by person ID.
   - Trains a Siamese network with `TripletMarginLoss`.
   - Uses a distance threshold to decide whether two signatures are from the same person.

2. **Environmental change detection**
   - Reuses the Siamese architecture for before/after image comparison.
   - Trains on paired images labeled as `Positive`, `Negative`, or `No_Change`.
   - Uses a custom distance-based decision process for change classification.

## Key components

- **Custom dataset classes** for generating triplets and image pairs on the fly.
- **`SimpleEmbeddingNetwork`** as the feature extractor/backbone.
- **`SiameseNetwork`** as a shared-weights wrapper for triplet or pair inference.
- **Training loops** for signature verification and change detection.
- **Visualization helpers** for inspecting samples and predictions.

## Training utilities

The notebook uses helpers from `training_functions.py`, including:

- `training_loop_signature(...)`
- `training_loop_change(...)`
- `evaluation_loop(...)`

## Helper utilities

`helper_utils.py` includes functions to:

- summarize the signature dataset
- show random signature pairs and triplets
- split datasets into train/validation subsets
- display validation predictions
- verify signature pairs
- work with the change detection dataset
- compute metrics and plot confusion matrices

## Dataset layout expected for signature verification

```text
./Signature_Verification_v5_v11/
├── Real/
│   ├── ID_01/
│   ├── ID_02/
│   └── ...
└── Fake/
    ├── ID_01/
    ├── ID_02/
    └── ...
```

Each `ID_*` folder should contain the corresponding real or fake signature images.

## Notes

- The notebook is designed for interactive use in Jupyter.
- Some utilities assume the dataset is already organized in the expected folder structure.
- The notebook uses normalized 224x224 RGB inputs.
