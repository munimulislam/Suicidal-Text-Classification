import sys
import json
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
from config import SEED, KAGGLE, SWMH, KAGGLE_PROCESSED, SWMH_PROCESSED, RESULTS_DIR, CHECKPOINT_DIR, KAGGLE_LABELS, SWMH_LABELS
from utils.dl_utils import load_glove_embeddings, build_vocab_and_matrix, tokenise, TextDataset, train_epoch, evaluate, save_loss_curve
from utils.metrics_utils import print_metrics, save_confusion_matrix

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
    Standalone CNN text classifier.

    Architecture:
        Embedding (GloVe 100d, fine-tuned)
        → Conv1D × 3 (kernel sizes 2, 3, 4) with ReLU
        → Global MaxPool per conv branch
        → Concatenate all branches
        → Dropout
        → Linear → num_labels

    The three kernel sizes act as n-gram detectors:
        kernel=2 → bigrams  ("kill myself", "want to")
        kernel=3 → trigrams ("want to die")
        kernel=4 → 4-grams  ("I want to die")
    """
    def __init__(self, vocab_size: int, embed_matrix: np.ndarray, num_labels: int):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embed_matrix),
            freeze=False,       # allow fine-tuning
            padding_idx=0       # <pad> gets zero gradient
        )

        # One Conv1D per kernel size
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=EMBED_DIM, out_channels=FILTERS, kernel_size=k) for k in KERNEL_SIZES]
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(FILTERS * len(KERNEL_SIZES), num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)             # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)         # (batch, embed_dim, seq_len)

        pooled = []

        for conv in self.convs:
            c = F.relu(conv(emb))           # (batch, filters, seq_len - k + 1)
            c = F.max_pool1d(c, c.size(2))  # (batch, filters, 1)
            c = c.squeeze(2)                # (batch, filters)
            pooled.append(c)

        # Concatenate all kernel outputs
        cat = torch.cat(pooled, dim=1)      # (batch, filters * num_kernels)
        out = self.dropout(cat)

        return self.fc(out)                 # (batch, num_labels)


def run_cnn(dataset_name: str):
    print(f"\n{'='*60}")
    print(f"CNN — {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── Paths ──────────────────────────────────────────────
    processed_dir = KAGGLE_PROCESSED if dataset_name == KAGGLE else SWMH_PROCESSED
    results_dir = RESULTS_DIR / dataset_name / "cnn"
    ckpt_dir = CHECKPOINT_DIR / f"cnn_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    # CNN uses lemmatized text (no stopwords, no punctuation)
    X_train_raw = train_df['text_lemmatized'].fillna('').astype(str).values
    X_val_raw = val_df['text_lemmatized'].fillna('').astype(str).values
    X_test_raw = test_df['text_lemmatized'].fillna('').astype(str).values

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    num_labels = len(np.unique(y_train))
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Classes: {num_labels} | Labels: {np.unique(y_train)}")

    # ── GloVe + Vocab ──────────────────────────────────────
    print("\n[2/6] Building vocabulary and embedding matrix...")
    glove = load_glove_embeddings(EMBED_DIM)
    vocab, embed = build_vocab_and_matrix(X_train_raw, glove, EMBED_DIM, MAX_VOCAB)
    del glove  # free memory — embeddings are now in matrix

    # ── Tokenise ───────────────────────────────────────────
    print("\n[3/6] Tokenising...")
    X_train = tokenise(X_train_raw, vocab)
    X_val = tokenise(X_val_raw, vocab)
    X_test = tokenise(X_test_raw, vocab)

    # ── DataLoaders ────────────────────────────────────────
    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────
    print("\n[4/6] Building CNN model...")

    model = CNNClassifier(len(vocab), embed, num_labels).to(device)
    class_weights = np.load(str(processed_dir / "class_weights.npy"))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    optimizer = Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Training ───────────────────────────────────────────
    print("\n[5/6] Training...")
    best_val_f1 = 0.0
    patience_ctr = 0
    train_losses = []
    val_losses = []
    ckpt_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, num_labels, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val Loss: {vl_loss:.4f} | "
              f"Val Macro F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_ctr += 1

            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    save_loss_curve(train_losses, val_losses, title=f"CNN Loss — {dataset_name.upper()}", out_path=results_dir / "loss_curve.png")

    # ── Test Evaluation ────────────────────────────────────
    print("\n[6/6] Evaluating best model on test set...")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_metrics, y_pred, _ = evaluate(model, test_loader, criterion, num_labels, device)

    print("Test metrics:")
    print_metrics(test_metrics)

    labels = (KAGGLE_LABELS if dataset_name == KAGGLE else SWMH_LABELS)
    save_confusion_matrix(y_test, y_pred, labels, title=f"Confusion Matrix — CNN ({dataset_name.upper()})", out_path=results_dir / "confusion_matrix.png")

    # ── Save Metrics ───────────────────────────────────────
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
        }
    }

    metrics_path = results_dir / "metrics.json"

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved {metrics_path}")
    
    print(f"\nCNN complete — {dataset_name.upper()}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(f"FNR: {test_metrics['fnr']:.4f}")

    return test_metrics


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='M03 — Standalone CNN text classifier')
    parser.add_argument('--dataset', choices=[KAGGLE, SWMH], default=None, help='Dataset to run. If not provided, runs both.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else [KAGGLE, SWMH]

    for ds in datasets:
        run_cnn(ds)