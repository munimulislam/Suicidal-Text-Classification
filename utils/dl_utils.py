import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import urllib.request
import zipfile
from config import GLOVE_DIR
from utils.metrics_utils import compute_metrics

# ─────────────────────────────────────────────────────────────
# GloVe
# ─────────────────────────────────────────────────────────────

def setup_glove(dim: int = 100) -> Path:
    glove_dir = Path(GLOVE_DIR)
    glove_dir.mkdir(parents=True, exist_ok=True)

    txt_file = glove_dir / f"glove.6B.{dim}d.txt"
    zip_file = glove_dir / "glove.6B.zip"

    if txt_file.exists():
        print(f"  GloVe found at {txt_file}")
        return txt_file

    if not zip_file.exists():
        __download_glove_zip(zip_file)

    __extract_glove_txt(zip_file, glove_dir, dim)
    
    return txt_file


def __download_glove_zip(path: Path):
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    print(f"Downloading GloVe embeddings (~822MB)...")
    print(f"Source: {url}")
    print(f"Destination: {path}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size

        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = '=' * int(pct // 2) + '||' * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:.1f}%", end='', flush=True)

    urllib.request.urlretrieve(url, path, reporthook=_progress)

    print()
    print(f"Download complete.")


def __extract_glove_txt(zip_path: Path, dest_path: Path, dim: int):
    with zipfile.ZipFile(zip_path, 'r') as z:
        target = f"glove.6B.{dim}d.txt"

        if target not in z.namelist():
            available = [n for n in z.namelist() if n.endswith('.txt')]
            raise ValueError(
                f"glove.6B.{dim}d.txt not found in zip.\n"
                f"Available: {available}"
            )
        
        z.extract(target, dest_path)

    print(f"Extraction Complete")


def load_glove_embeddings(embed_dim: int = 100) -> dict:
    glove_path = setup_glove(embed_dim)

    embeddings = {}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    
    return embeddings


def build_vocab_and_matrix(texts: np.ndarray, glove: dict, embed_dim: int, max_vocab: int) -> tuple:
    from collections import Counter

    counter = Counter()
    vocab = {'<pad>': 0, '<unk>': 1}

    for text in texts:
        counter.update(str(text).lower().split())

    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    # Random init — GloVe hits will overwrite
    matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = np.zeros(embed_dim)  # <pad> = zeros
    hits = 0

    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1

    coverage = hits / len(vocab)
    print(f"  Vocab size: {len(vocab):,} | GloVe coverage: {coverage:.1%}")

    return vocab, matrix


def tokenise(texts: np.ndarray, vocab: dict, max_len: int = 256) -> np.ndarray:
    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']
    result = []

    for text in texts:
        tokens  = str(text).lower().split()[:max_len]
        indices = [vocab.get(t, unk_idx) for t in tokens]
        # Right-pad to max_len
        indices += [pad_idx] * (max_len - len(indices))
        result.append(indices)

    return np.array(result, dtype=np.int64)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised integer sequences and labels.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def save_loss_curve(train_losses: list,
                    val_losses: list,
                    title: str,
                    out_path: Path):
    """Save training/validation loss curve as PNG."""
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


# ─────────────────────────────────────────────────────────────
# Training / Evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module,
                loader,
                optimizer,
                criterion,
                device: torch.device) -> float:
    """
    Run one training epoch.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        # Gradient clipping — prevents exploding gradients in BiLSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model: nn.Module,
             loader,
             criterion,
             num_labels: int,
             device: torch.device) -> tuple:
    """
    Evaluate model on a DataLoader.

    Returns:
        avg_loss  : float
        metrics   : dict
        y_pred    : np.ndarray of predicted labels
        y_prob    : np.ndarray of predicted probabilities
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_prob   = np.array(all_probs)
    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(y_true, y_pred, y_prob, num_labels)

    return avg_loss, metrics, y_pred, y_prob