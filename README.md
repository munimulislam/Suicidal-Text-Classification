# DL-Suicidal-Text-Classification

Deep learning models for classifying suicidal ideation in social-media text.

## Overview

This repository explores deep-learning approaches (baselines → LSTM → Transformer-based models) for detecting suicidal ideation in Reddit-sourced text. The goal is to support research in mental-health NLP and early-warning screening.

> **Disclaimer:** This project is for research and educational purposes only. It is **not** a clinical tool and must not be used to make diagnostic or treatment decisions. If you or someone you know is in crisis, please contact a local emergency service or a suicide-prevention hotline.

## Datasets

Two datasets are used. See [DATA.md](DATA.md) for full access instructions.

| Dataset                                                                                | Access                                                                         | Location       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------- |
| [Kaggle SuicideWatch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | Public                                                                         | `data/kaggle/` |
| [SWMH (Zenodo)](https://zenodo.org/records/6476179)                                    | Restricted — institutional-email request required; **cannot be redistributed** | `data/swmh/`   |

Raw data files are gitignored and must be downloaded locally.

## Repository Structure

.
├── data/ # Raw and processed datasets (gitignored)
│ ├── kaggle/ # Kaggle SuicideWatch CSV
│ └── swmh/ # SWMH files (restricted)
├── notebooks/ # Jupyter notebooks for EDA, preprocessing, training, evaluation
├── checkpoints/ # Saved model checkpoints and artifacts
├── results/ # Metrics, figures, and experiment outputs
├── DATA.md # Dataset access instructions
└── requirements.txt # Python dependencies

## Setup

```bash
git clone https://github.com/munimulislam/DL-Suicidal-Text-Classification.git
cd DL-Suicidal-Text-Classification

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
Then download the datasets as described in DATA.md.
```

### Preprocessing

```bash
python scripts/preprocess.py                    # process both datasets
python scripts/preprocess.py --dataset kaggle   # Kaggle only
python scripts/preprocess.py --dataset swmh     # SWMH only
```

Produces `text_clean` (for transformers) and `text_lemmatized` (for CNN/BiLSTM) columns.

### Model Training

```bash
python scripts/logistic_regression.py --dataset [kaggle/swmh]
```

```bash
python scripts/cnn.py --dataset [kaggle/swmh]
python scripts/bi_lstm.py --dataset [kaggle/swmh]
```

```bash
python scripts/train_transformer.py --model [distilbert/mentalbert/roberta] --dataset [kaggle/swmh]
```

### Cross-Dataset Transfer Test

```bash
python scripts/transfer_test.py --model [distilbert/mentalbert/roberta]
```

### XAI Analysis

```bash
python scripts/xai_analysis.py --model [distilbert/mentalbert/roberta]  --dataset [kaggle/swmh]
```

### Results Summary

```bash
python scripts/results_summary.py
```

## Running on Kaggle (for GPU)

```python
# Cell 1 — Clone repo
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
import os
os.chdir('/kaggle/working/YOUR_REPO')

# Cell 2 — Install dependencies
!pip install -r requirements.txt -q

# Cell 3 — Run training
!python scripts/train_transformer.py --model roberta --dataset kaggle
```

Upload processed CSVs as a Kaggle Dataset. Add it as input to your notebook.

---

## Results Summary

| Model       | Kaggle F1  | Kaggle FNR | SWMH F1    | SWMH FNR |
| ----------- | ---------- | ---------- | ---------- | -------- |
| TF-IDF + LR | 0.9400     | 0.0673     | 0.6687     | 0.3194   |
| CNN         | 0.9482     | 0.0579     | 0.6509     | 0.3274   |
| BiLSTM      | 0.9522     | 0.0534     | 0.6740     | 0.3206   |
| DistilBERT  | 0.9785     | 0.0239     | 0.7034     | 0.2934   |
| MentalBERT  | 0.9805     | 0.0208     | 0.7244     | 0.2637   |
| **RoBERTa** | **0.9915** | **0.0097** | **0.7270** | 0.2692   |

---

## Deployment

See the `deployment/` directory for the standalone FastAPI + HTMX web application.
