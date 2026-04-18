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
├── models/ # Saved model checkpoints and artifacts
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
