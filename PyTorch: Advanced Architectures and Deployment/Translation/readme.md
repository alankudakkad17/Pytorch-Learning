# Translation

This folder contains materials for building an English ↔ Target-language translation system using transformer encoder-decoder models (PyTorch).

Contents
- C3M3_Assignment.ipynb — Programming assignment / notebook that walks through dataset preparation, building encoder/decoder/encoder-decoder models, training and evaluation.
- helper_utils.py — Utility functions for loading datasets, tokenization, normalization, and training helpers.
- unittests.py — Grading and unit tests used by the assignment to validate the Encoder, Decoder, and EncoderDecoder implementations.

Dataset
- The notebook expects parallel English↔{language} files to be available in a `languages/` directory at the repository root. The supported target languages and filenames are defined in `helper_utils.LANGUAGE_PAIRS`.
- If you have the raw datasets, place them or the zipped archives (e.g. `deu-eng.zip`) into `languages/` and the helper will extract/load them.

Requirements
- Python 3.8+ recommended
- PyTorch
- tqdm
- matplotlib

You can install common requirements with:

```bash
pip install torch tqdm matplotlib
```

Running the assignment
1. Open `C3M3_Assignment.ipynb` with Jupyter / JupyterLab / Colab.
2. Run the cells in order. The notebook will prompt you to select a target language and will guide you through preprocessing, model implementation exercises, and training.
3. To run the unit tests that validate your model classes, the notebook imports `unittests.py` (used in the assignment environment).

Notes
- The assignment cells that accept your implementation are marked as editable/graded in the notebook. Avoid changing those locations unless you are submitting your solution.
- The helper tokenizer preserves contractions and handles language-specific characters (accents, umlauts, Cyrillic) where appropriate.

