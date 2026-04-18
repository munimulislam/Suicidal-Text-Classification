import numpy as np
import pandas as pd
import argparse
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEED, KAGGLE, SWMH, KAGGLE_PROCESSED, SWMH_PROCESSED, RESULTS_DIR, KAGGLE_LABELS, SWMH_LABELS

def compute_metrics(y_true, y_pred, y_prob, num_labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    if num_labels == 2:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    if num_labels == 2:
        cm = confusion_matrix(y_true, y_pred)
        fn = cm[1, 0]
        tp = cm[1, 1]
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        cm = confusion_matrix(y_true, y_pred)
        fnr_per_class = []

        for i in range(num_labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fnr_per_class.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
            
        metrics['fnr'] = np.mean(fnr_per_class)

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    _, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    plt.title('Confusion Matrix — TF-IDF + Logistic Regression')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved : {out_path}")


def print_metrics(metrics: dict):
    print(f"\n  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*30}")

    for k, v in metrics.items():
        print(f"  {k:<20} {v:>8.4f}")


def run_logistic(dataset_name: str):
    print(f"\n{'='*60}")
    print(f"TF-IDF + Logistic Regression — {dataset_name.upper()}")
    print(f"{'='*60}")

    processed_dir = KAGGLE_PROCESSED if dataset_name == KAGGLE else SWMH_PROCESSED
    results_dir = RESULTS_DIR / dataset_name / "logistic"
    results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")

    X_train = train['text_lemmatized'].fillna('').astype(str)
    X_val = val['text_lemmatized'].fillna('').astype(str)
    X_test = test['text_lemmatized'].fillna('').astype(str)

    y_train = train['label'].values
    y_val = val['label'].values
    y_test = test['label'].values

    num_labels = len(np.unique(y_train))
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Classes: {num_labels} | Labels: {np.unique(y_train)}")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2, sublinear_tf=True)
    X_train_tfidf = tfidf.fit(X_train).transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")

    clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced', C=1.0,solver='saga')
    clf.fit(X_train_tfidf, y_train)
    print("\nTraining complete.\n")

    print("\nEvaluating on validation set...")
    y_val_pred = clf.predict(X_val_tfidf)
    y_val_prob = clf.predict_proba(X_val_tfidf)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, num_labels)
    print("  Validation metrics:")
    print_metrics(val_metrics)

    print("\nEvaluating on test set...")
    y_test_pred = clf.predict(X_test_tfidf)
    y_test_prob = clf.predict_proba(X_test_tfidf)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, num_labels)
    print("  Test metrics:")
    print_metrics(test_metrics)

    results = {
        "model": "tfidf_lr",
        "dataset": dataset_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "tfidf_params": {
            "ngram_range": "(1,2)",
            "max_features": 100_000,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "lr_params": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 1000,
        }
    }

    metrics_path = results_dir / "metrics.json"

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Metrics saved at: {metrics_path}")

    labels = KAGGLE_LABELS if dataset_name == KAGGLE else SWMH_LABELS
    save_confusion_matrix(y_test, y_test_pred, labels, results_dir / "confusion_matrix.png")

    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(f"FNR: {test_metrics['fnr']:.4f}")

    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='TF-IDF + Logistic Regression baseline')
    parser.add_argument('--dataset', type=str, choices=[KAGGLE, SWMH], default=None, help='Dataset to run. If not provided, runs both.')

    return parser.parse_args()


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    args = parse_args()

    if args.dataset == KAGGLE:
        run_logistic(KAGGLE)
    elif args.dataset == SWMH:
        run_logistic(SWMH)
    else:
        run_logistic(KAGGLE)
        run_logistic(SWMH)