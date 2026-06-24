# Parkinson's Disease Detection using EEG Signals and Machine Learning

## Overview

This project develops a complete machine learning pipeline for Parkinson's Disease (PD) detection using resting-state EEG recordings and participant metadata.

The pipeline automatically:

* Preprocesses raw EEG recordings (.bdf)
* Extracts EEG frequency-domain features
* Generates participant metadata
* Creates a merged research dataset
* Trains multiple machine learning models
* Performs Monte Carlo validation
* Detects overfitting and underfitting
* Produces research-grade visualizations
* Saves trained models and evaluation reports

The objective is to investigate whether EEG biomarkers combined with demographic and cognitive information can help distinguish Parkinson's Disease patients from Healthy Controls.

---

## Dataset

### EEG Dataset

The project uses EEG recordings stored as:

```text
data/raw/*.bdf
```

Each EEG file belongs to either:

* Healthy Control (HC)
* Parkinson's Disease (PD)

### Metadata

Participant metadata is stored in:

```text
data/raw/participants.tsv
```

Metadata includes:

| Feature          | Description                   |
| ---------------- | ----------------------------- |
| age              | Participant age               |
| gender           | Male/Female                   |
| hand             | Handedness                    |
| MMSE             | Mini Mental State Examination |
| NAART            | National Adult Reading Test   |
| disease_duration | Years since diagnosis         |
| rl_deficits      | Motor deficit information     |
| notes            | Clinical notes                |

---

## Project Structure

```text
eeg/

│
├── data/
│   ├── raw/
│   │   ├── participants.tsv
│   │   └── *.bdf
│   │
│   └── processed/
│       ├── metadata.csv
│       ├── eeg_features.csv
│       └── final_dataset.csv
│
├── outputs/
│   ├── plots/
│   └── models/
│
├── src/
│   ├── config.py
│   ├── dataset_builder.py
│   ├── feature_builder.py
│   ├── merge_dataset.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## EEG Feature Engineering

### Frequency Band Features

| Feature | Frequency Range |
| ------- | --------------- |
| Delta   | 1–4 Hz          |
| Theta   | 4–8 Hz          |
| Alpha   | 8–12 Hz         |
| Beta    | 12–30 Hz        |

### Derived Band Ratios

* Delta / Theta Ratio
* Alpha / Theta Ratio
* Beta / Alpha Ratio

### Statistical Features

* Mean
* Standard Deviation
* Variance
* Maximum
* Minimum

### Total EEG Features

```text
12 EEG Features
```

---

## Metadata Features

The final dataset additionally includes:

* Age
* Gender
* Handedness
* MMSE
* NAART
* Disease Duration
* Season Cohort Proxy

---

## Season Cohort Proxy

Because acquisition dates were not available in the original dataset, a season cohort proxy is generated from participant ordering.

The proxy creates temporal cohort groups:

```text
Winter
Spring
Summer
Autumn
```

This feature is included as an exploratory temporal covariate and does not represent true collection dates.

---

## Machine Learning Models

The following classifiers are trained:

### Random Forest

```python
RandomForestClassifier
```

### Support Vector Machine

```python
SVC
```

### Logistic Regression

```python
LogisticRegression
```

### K-Nearest Neighbors

```python
KNeighborsClassifier
```

---

## Validation Strategy

### Train/Test Split

```text
80% Training
20% Testing
```

### Monte Carlo Validation

```text
20 Random Resampling Runs
```

Metrics collected:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Automatic Model Auditing

The evaluation engine automatically flags:

### Overfitting

Large train-test accuracy gap.

### Underfitting

Low train and test performance.

### High Variance

Unstable Monte Carlo performance.

### Low Precision

Excessive false positives.

### Low Recall

Excessive false negatives.

### Weak Class Separation

Poor ROC-AUC performance.

---

## Visualizations

The pipeline automatically generates:

### Dataset Analysis

* Label Distribution
* Gender Distribution
* Season Distribution
* Age Histogram
* MMSE Histogram
* NAART Histogram
* Disease Duration Histogram
* EEG Band Histograms
* Correlation Heatmap
* Pairplot
* PCA Projection

### EEG Feature Analysis

* Boxplots
* Violin Plots
* Distribution Curves

### Model Analysis

* Training Accuracy
* Testing Accuracy
* Generalization Gap
* Monte Carlo Comparison

All plots are saved to:

```text
outputs/plots/
```

---

## Running the Pipeline

### Step 1

Create environment:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

### Step 2

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 3

Run complete pipeline:

```bash
python main.py
```

---

## Outputs

### Processed Data

```text
data/processed/
```

### Trained Models

```text
outputs/models/
```

### Evaluation Reports

```text
outputs/models/monte_carlo_results.csv
```

```text
outputs/models/model_comparison.csv
```

### Visualizations

```text
outputs/plots/
```

---

## Research Goal

This project explores whether EEG-derived biomarkers combined with cognitive and demographic variables can provide useful predictive signals for Parkinson's Disease classification.

The pipeline is designed as a reproducible machine learning workflow suitable for:

* Biomedical Signal Processing
* EEG Analytics
* Machine Learning Research
* Healthcare AI Projects
* Portfolio Demonstrations

---

## Author

Hitansh Purohit

Machine Learning • AI Engineering • Biomedical Signal Processing
