# EEG Sleep Stage Classification

Automated sleep stage classification using EEG signals from the Sleep-EDF dataset, implementing Discrete Wavelet Transform (DWT) feature extraction with AdaBoost and MLP (Multi-Layer Perceptron) classifiers, compared across 5-class, 4-class, 3-class, and 2-class sleep staging problems.

**Associated Publication:** IEEE ICONAT 2025  
**DOI:** [10.1109/ICONAT66879.2025.11362450](https://doi.org/10.1109/ICONAT66879.2025.11362450)

---

## Overview

This project implements an independent exploration of automated EEG-based sleep stage classification, inspired by the methodology described in the associated IEEE publication. Raw single-channel EEG signals are segmented into 30-second epochs, decomposed using Discrete Wavelet Transform (DWT), and classified into sleep stages using two different machine learning approaches: AdaBoost and a Multi-Layer Perceptron (MLP) neural network.

## Dataset

**Sleep-EDF Database** (original, 8-recording version) — PhysioNet  
https://physionet.org/content/sleep-edf/1.0.0/

Sleep stages considered (depending on class granularity):
- Wake (W)
- Stage 1 (S1) — light sleep
- Stage 2 (S2) — intermediate sleep
- Stage 3/4 (S3/S4) — merged as deep sleep
- REM — rapid eye movement sleep

## Methodology

1. **Signal Loading** — Raw `.rec` (EDF format) files loaded via MNE, with the Pz-Oz channel selected for analysis
2. **Hypnogram Parsing** — Sleep stage labels extracted from the original Rechtschaffen & Kales (R&K) byte-encoded `.hyp` format
3. **Epoching** — Signals segmented into 30-second epochs (3,000 samples at 100 Hz)
4. **Preprocessing** — Bandpass filtering (0.5–40 Hz) and z-score normalization, with artifact rejection based on amplitude thresholding
5. **Feature Extraction** — Discrete Wavelet Transform (db4 wavelet, 4 decomposition levels), extracting energy, RMS, mean, standard deviation, entropy, skewness, and kurtosis from each sub-band
6. **Feature Selection** — ANOVA-based feature selection (`SelectKBest`, top 20 features) to retain only the most statistically significant features
7. **Classification** — Two models trained and compared:
   - AdaBoost with shallow decision trees, combined with random undersampling for class imbalance
   - MLP (Multi-Layer Perceptron) neural network, using the same feature selection and undersampling pipeline
8. **Evaluation** — Leave-One-Subject-Out cross-validation (each fold trains on 7 subjects and tests on 1 entirely unseen subject)

## Results

| Classification Task | AdaBoost Accuracy | MLP Accuracy |
|---|---|---|
| 5-Class (W, S1, S2, S3/4, REM) | 21.5% | 31.6% |
| 4-Class (W, Light, Deep, REM) | 28.6% | 36.4% |
| 3-Class (W, NREM, REM) | 25.9% | 39.5% |
| 2-Class (W, Sleep) | 48.6% | 52.4% |

**MLP consistently outperforms AdaBoost across every classification scenario**, consistent with the original paper's choice of a neural-network-based approach over simpler tree-based boosting.

## Methodology Note — Why These Numbers Differ from the Published Paper

This implementation deliberately uses **Leave-One-Subject-Out cross-validation**, where each fold trains on data from 7 subjects and tests on a completely unseen 8th subject. This is a substantially stricter evaluation protocol than a standard shuffled train/test split, since EEG signal patterns vary meaningfully between individuals — the model cannot rely on having seen similar data from the same subject during training.

The associated IEEE paper reports accuracies of 86% (5-class) up to 97% (2-class), which likely reflects a less strict evaluation setup. The gap observed here highlights an important practical lesson in EEG-based machine learning: **subject-independent generalization is a substantially harder problem than within-subject classification**, and evaluation protocol choice significantly impacts reported performance.

Additional contributing factors:
- Severe class imbalance (Wake: 12,532 samples vs. Stage 1: 338 samples)
- Limited subject diversity (8 total recordings)
- Random undersampling discards a large portion of majority-class data each fold

## Tech Stack

Python · NumPy · MNE · PyWavelets · SciPy · Scikit-learn · Imbalanced-learn · Matplotlib · tqdm

## How to Run

```bash
pip install -r requirements.txt
```

Update `DATA_DIR` in `main.py` to point to your local Sleep-EDF dataset folder, then run:

```bash
python main.py
```

## Project Structure

```
eeg-sleep-stage-classification/
├── main.py              # Full pipeline: loading, preprocessing, features, training, evaluation
├── requirements.txt     # Python dependencies
├── .gitignore            # Excludes dataset files and generated artifacts
└── README.md
```

## Future Improvements

- Evaluate on the larger Sleep-EDF Expanded dataset (78 subjects) for improved generalization
- Implement Ant Colony Optimization (ACO) for feature selection, matching the original paper's approach
- Explore class-weighted loss functions as an alternative to random undersampling
- Add deep learning approaches (CNN/LSTM) for end-to-end feature learning directly from raw signals
