import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay)    

def compute_metrics(y_true, y_pred, y_prob, num_labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    is_binary_clf = num_labels == 2

    if is_binary_clf:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    cm = confusion_matrix(y_true, y_pred)

    if is_binary_clf:
        fn = cm[1, 0]
        tp = cm[1, 1]
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fnr_per_class = []

        for i in range(num_labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fnr_per_class.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
            
        metrics['fnr'] = np.mean(fnr_per_class)

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, title, out_path) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    _, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved at: {out_path}")


def print_metrics(metrics: dict):
    print(f"\n  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*30}")

    for k, v in metrics.items():
        print(f"  {k:<20} {v:>8.4f}")


def save_loss_curve(train_losses: list, val_losses: list, title: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Val Loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Loss curve saved → {out_path}")

def save_result(file_path: Path, result: dict):
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2)