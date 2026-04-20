import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import SEED
from utils.metrics_utils import compute_metrics

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "mentalbert": "mental/mental-bert-base-uncased",
    "roberta": "roberta-base",
}


class TransformerDataset(Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(
    model, loader, optimizer, scheduler, criterion, device: torch.device, scaler=None
) -> float:
    """
    One training epoch for transformer models.

    Returns average training loss.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, num_labels: int, device: torch.device) -> tuple:
    """
    Evaluate transformer model on a DataLoader.

    Returns:
        avg_loss : float
        metrics  : dict
        y_pred   : np.ndarray
        y_prob   : np.ndarray
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_labels)

    return avg_loss, metrics, y_pred, y_prob
