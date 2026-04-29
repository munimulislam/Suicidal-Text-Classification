import sys
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SEED,
    KAGGLE,
    SWMH,
    KAGGLE_PROCESSED,
    SWMH_PROCESSED,
    RESULTS_DIR,
    CHECKPOINT_OUT_DIR,
    KAGGLE_LABELS,
    SWMH_LABELS,
)
from utils.dl_utils import (
    load_tokenized_data_with_embedding,
    load_glove_embeddings,
    build_vocab_and_matrix,
    TextDataset,
    train_epoch,
    evaluate,
)
from utils.metrics_utils import (
    print_metrics,
    save_confusion_matrix,
    save_loss_curve,
    save_result,
)

EMBED_DIM = 100
MAX_VOCAB = 50_000
MAX_SEQ_LEN = 256
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
PATIENCE = 3
DROPOUT = 0.5
FILTERS = 128
KERNEL_SIZES = [2, 3, 4]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class CNNClassifier(nn.Module):
    """
    Architecture:
        Embedding (GloVe 100d, fine-tuned)
        → Conv1D × 3 (kernel sizes 2, 3, 4) with ReLU
        → Global MaxPool per conv branch
        → Concatenate all branches
        → Dropout
        → Linear → num_labels
    """

    def __init__(self, vocab_size: int, embed_matrix: np.ndarray, num_labels: int):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embed_matrix),
            freeze=False,
            padding_idx=0,  # <pad> gets zero gradient
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=EMBED_DIM, out_channels=FILTERS, kernel_size=k)
                for k in KERNEL_SIZES
            ]
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(FILTERS * len(KERNEL_SIZES), num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        pooled = []

        for conv in self.convs:
            c = F.relu(conv(emb))  # (batch, filters, seq_len - k + 1)
            c = F.max_pool1d(c, c.size(2))  # (batch, filters, 1)
            c = c.squeeze(2)  # (batch, filters)
            pooled.append(c)

        cat = torch.cat(pooled, dim=1)  # (batch, filters * num_kernels)
        out = self.dropout(cat)

        return self.fc(out)  # (batch, num_labels)


def run_cnn(dataset_name: str):
    print(f"CNN — {dataset_name.upper()}")

    processed_dir = KAGGLE_PROCESSED if dataset_name == KAGGLE else SWMH_PROCESSED
    results_dir = RESULTS_DIR / dataset_name / "cnn"
    ckpt_dir = CHECKPOINT_OUT_DIR / f"cnn_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, embed = (
        load_tokenized_data_with_embedding(dataset_name, EMBED_DIM, MAX_VOCAB)
    )

    train_loader = DataLoader(
        TextDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TextDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TextDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    num_labels = len(np.unique(y_train))
    class_weights = np.load(str(processed_dir / "class_weights.npy"))

    model = CNNClassifier(len(vocab), embed, num_labels).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )
    optimizer = Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    print("\nTraining...")
    best_val_f1 = 0.0
    patience_ctr = 0
    train_losses = []
    val_losses = []
    ckpt_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, num_labels, device
        )

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
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

            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    save_loss_curve(
        train_losses,
        val_losses,
        title=f"CNN Loss — {dataset_name.upper()}",
        out_path=results_dir / "loss_curve.png",
    )

    print("\nEvaluating best model on test set...")
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
        title=f"Confusion Matrix — CNN ({dataset_name.upper()})",
        out_path=results_dir / "confusion_matrix.png",
    )

    results = {
        "model": "cnn",
        "dataset": dataset_name,
        "test_metrics": test_metrics,
        "hyperparams": {
            "embed_dim": EMBED_DIM,
            "max_seq_len": MAX_SEQ_LEN,
            "cnn_filters": FILTERS,
            "kernel_sizes": KERNEL_SIZES,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_run": len(train_losses),
            "patience": PATIENCE,
        },
    }

    result_path = results_dir / "metrics.json"
    save_result(result_path, results)

    print(f"\nCNN complete — {dataset_name.upper()}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="M03 — Standalone CNN text classifier")
    parser.add_argument(
        "--dataset",
        choices=[KAGGLE, SWMH],
        default=None,
        help="Dataset to run. If not provided, runs both.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [args.dataset] if args.dataset else [KAGGLE, SWMH]

    for ds in datasets:
        run_cnn(ds)


if __name__ == "__main__":
    main()
