# 🐦 BIRDCLEF+ 2026

### Acoustic Species Identification in the Pantanal, South America

Machine Learning & Deep Learning project developed for the **BirdCLEF+ 2026 Kaggle Competition**, focused on identifying wildlife species from passive acoustic monitoring recordings.

## 📌 Project Overview

The project uses **deep learning and audio signal processing** to classify wildlife species from 5-second audio recordings.

The core approach converts audio into **Mel Spectrograms** and processes them using pretrained **EfficientNet** architectures.

**Best Public Leaderboard Score: `0.738 Macro ROC-AUC`**
**Baseline Score: `0.624` → **`18.2% improvement`**

## 📊 Dataset

* **35,549** labeled training audio files
* **206** species in training data
* **234** species in final submission
* **32,000 Hz** sample rate
* **5-second** audio windows
* Wildlife groups: Birds, Amphibians, Insects, Mammals & Reptiles

## 🧠 Approach

```text
Audio Recording
      ↓
5-Second Audio Window
      ↓
Mel Spectrogram
      ↓
Normalization
      ↓
Data Augmentation
      ↓
EfficientNet
      ↓
Sigmoid
      ↓
234 Species Probabilities
```

### Audio Processing

* Audio loaded at **32 kHz**
* Padded/trimmed to **5 seconds**
* **128 Mel bins**
* Frequency range: **20 Hz – 16 kHz**
* Power spectrogram converted to decibels
* Mean-Std normalization

### Augmentation

* Noise Injection
* Time Shift
* Gain Change
* Frequency Masking
* Time Masking / SpecAugment
* Mixup

## 🏗️ Model Experiments

| Version | Architecture        | Epochs | Key Features                |     Score |
| ------- | ------------------- | -----: | --------------------------- | --------: |
| V1      | EfficientNet-B0     |      3 | Baseline                    |     0.624 |
| V2      | EfficientNet-B2     |     10 | Full Data + Augmentation    |     0.724 |
| **V3**  | **EfficientNet-B4** | **15** | **Mixup + SpecAugment**     | **0.738** |
| V4      | EfficientNet-B4     |     25 | Extended Training           |     0.721 |
| V5      | EfficientNetV2-S    |     25 | Strong Augmentation         |     0.719 |
| V6      | EfficientNetV2-M    |     15 | Custom Head + Warm Restarts |     0.723 |

## ⚙️ Training

**Loss:** `BCEWithLogitsLoss`
**Optimizer:** `AdamW`
**Learning Rate:** `3e-4`
**Weight Decay:** `1e-4`
**Scheduler:** `CosineAnnealingLR`
**Gradient Clipping:** `max_norm=1.0`

Training was performed using **Kaggle T4 ×2 GPUs**, while inference was designed for **CPU-only execution**.

## 📈 Key Results

| Metric               |     Result |
| -------------------- | ---------: |
| Initial Score        |      0.624 |
| Best Score           |  **0.738** |
| Absolute Improvement | **+0.114** |
| Relative Improvement |  **18.2%** |
| Models Experimented  |         6+ |
| Submissions          |        15+ |
| Active Development   |   ~13 days |

## 🔬 Key Findings

* EfficientNet-B0 → B2 → B4 progressively improved performance.
* Using all **35,549 training samples** improved generalization.
* **Mixup** produced one of the largest performance improvements.
* **SpecAugment** helped with overfitting.
* Longer training without sufficient regularization resulted in overfitting.
* EfficientNetV2-S did not outperform EfficientNet-B4 in these experiments.

## 🚀 Future Improvements

* Google Perch / BirdNET pretrained audio models
* Proper cross-validation
* Training with soundscape data
* Ensemble of multiple models
* Pseudo-labeling
* Site-hour ecological priors for post-processing

## 🛠️ Tech Stack

`Python 3.12` · `PyTorch` · `timm` · `librosa` · `NumPy` · `pandas` · `scikit-learn` · `Kaggle`

## 📁 Project Structure

```text
birdclef-2026/
│
├── notebooks/
│   ├── birdclef-2026-starter.ipynb
│   └── birdclef-2026-training.ipynb
│
├── src/
├── models/
├── configs/
├── submissions/
├── requirements.txt
└── README.md
```

## 👥 Team

* **Hammad Ahmed** — `2023-SE-01`
* **Aman Tariq** — `2023-SE-29`
* **Aina Yousaf** — `2023-SE-32`


**Submitted to:** Ahmed Khwaja
**Kaggle Username:** `pythonophile`
**Platform:** Kaggle Competitions

## 📚 References

1. [BirdCLEF+ 2026 — Kaggle](https://www.kaggle.com/competitions/birdclef-2026)
2. Tan, M. & Le, Q. — *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*
3. Park, D. S. et al. — *SpecAugment*
4. Zhang, H. et al. — *Mixup: Beyond Empirical Risk Minimization*
5. McFee, B. et al. — *librosa: Audio and Music Signal Analysis in Python*
6. [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

### ⭐ BIRDCLEF+ 2026

**Deep Learning • Audio Classification • Wildlife Monitoring • Computer Vision • Kaggle**
