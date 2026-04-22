import sys
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SEED,
    KAGGLE,
    SWMH,
    KAGGLE_PROCESSED,
    SWMH_PROCESSED,
    KAGGLE_LABELS,
    SWMH_LABELS,
    RESULTS_DIR,
    CHECKPOINT_OUT_DIR,
)

from utils.transformer_utils import (
    MODEL_REGISTRY,
    TransformerDataset,
    train_epoch,
    evaluate,
)

from utils.metrics_utils import (
    print_metrics,
    save_confusion_matrix,
    save_loss_curve,
    save_result,
)

MAX_LEN = 256
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
PATIENCE = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


def run_transformer(model_name: str, dataset_name: str):
    hf_model_id = MODEL_REGISTRY[model_name]

    print(f"{model_name.upper()} — {dataset_name.upper()}")

    processed_dir = KAGGLE_PROCESSED if dataset_name == KAGGLE else SWMH_PROCESSED
    results_dir = RESULTS_DIR / dataset_name / model_name
    ckpt_dir = CHECKPOINT_OUT_DIR / f"{model_name}_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    X_train = train_df["text_clean"].fillna("").astype(str).tolist()
    X_val = val_df["text_clean"].fillna("").astype(str).tolist()
    X_test = test_df["text_clean"].fillna("").astype(str).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    num_labels = len(np.unique(y_train))
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes: {num_labels} | Labels: {np.unique(y_train)}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Max length: {MAX_LEN}")

    X_train_enc = tokenizer(
        X_train, max_length=MAX_LEN, padding="max_length", truncation=True
    )

    X_val_enc = tokenizer(
        X_val,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )

    X_test_enc = tokenizer(
        X_test,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )

    train_dataset = TransformerDataset(X_train_enc, y_train.tolist())
    val_dataset = TransformerDataset(X_val_enc, y_val.tolist())
    test_dataset = TransformerDataset(X_test_enc, y_test.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = np.load(str(processed_dir / "class_weights.npy"))

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # handles classifier head size mismatch
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01, eps=1e-8)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")

    print(f"\nTraining...")
    best_val_f1 = 0.0
    patience_ctr = 0
    train_losses = []
    val_losses = []
    ckpt_path = ckpt_dir / "best_model.pt"

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler=scaler
        )

        vl_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, num_labels, device
        )

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"  Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {vl_loss:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        has_improved = val_metrics["macro_f1"] > best_val_f1

        if has_improved:
            patience_ctr = 0
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")

            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    save_loss_curve(
        train_losses,
        val_losses,
        title=f"{model_name.upper()} Loss — {dataset_name.upper()}",
        out_path=results_dir / "loss_curve.png",
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_metrics, y_pred, _ = evaluate(
        model, test_loader, criterion, num_labels, device
    )

    print_metrics(test_metrics)

    labels = KAGGLE_LABELS if dataset_name == KAGGLE else SWMH_LABELS
    save_confusion_matrix(
        y_test,
        y_pred,
        labels,
        title=f"Confusion Matrix — {model_name.upper()} ({dataset_name.upper()})",
        out_path=results_dir / "confusion_matrix.png",
    )

    results = {
        "model": model_name,
        "hf_model_id": hf_model_id,
        "dataset": dataset_name,
        "test_metrics": test_metrics,
        "hyperparams": {
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_run": len(train_losses),
            "patience": PATIENCE,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
        },
    }

    metrics_path = results_dir / "metrics.json"
    save_result(metrics_path, results)

    print(f"\n{model_name.upper()} complete — {dataset_name.upper()}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tier 3 — Standalone Transformer Fine-tuning"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Transformer model to fine-tune",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[KAGGLE, SWMH],
        default=None,
        help="Dataset to train on. If not provided, trains on both.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else [KAGGLE, SWMH]

    for ds in datasets:
        run_transformer(args.model, ds)
