import argparse
import random
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHECKPOINT_IN_DIR,
    MAX_LEN,
    SEED,
    KAGGLE,
    SWMH,
    KAGGLE_PROCESSED,
    SWMH_PROCESSED,
    RESULTS_DIR,
)
from utils.transformer_utils import MODEL_REGISTRY
from utils.metrics_utils import compute_metrics

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SWMH_TO_BINARY = {
    0: 1,  # SuicideWatch - suicidal
    1: 0,  # depression - non-suicidal
    2: 0,  # anxiety - non-suicidal
    3: 0,  # bipolar - non-suicidal
    4: 0,  # offmychest - non-suicidal
}


def load_model(model_name: str, dataset_name: str, num_labels: int):
    ckpt_path = CHECKPOINT_IN_DIR / f"{model_name}_{dataset_name}" / "best_model.pt"
    hf_model_id = MODEL_REGISTRY[model_name]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def predict_probs(model, tokenizer, texts: list, batch_size: int = 32) -> np.ndarray:
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            ).logits

        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)


def kaggle_to_swmh(model_name: str):
    df = pd.read_csv(SWMH_PROCESSED / "test.csv")
    X = df["text_clean"].fillna("").astype(str).tolist()
    y = df["label"].map(SWMH_TO_BINARY).values

    model, tokenizer = load_model(model_name, KAGGLE, num_labels=2)

    probs = predict_probs(model, tokenizer, X)
    preds = np.argmax(probs, axis=1)

    metrics = compute_metrics(y, preds, probs, num_labels=2)

    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")

    return metrics


def swmh_to_kaggle(model_name: str):
    df = pd.read_csv(KAGGLE_PROCESSED / "test.csv")
    X = df["text_clean"].fillna("").astype(str).tolist()
    y = df["label"].values

    model, tokenizer = load_model(model_name, SWMH, num_labels=5)

    probs = predict_probs(model, tokenizer, X)
    preds = (np.argmax(probs, axis=1) == 0).astype(int)
    probs_binary = np.column_stack([1 - probs[:, 0], probs[:, 0]])
    metrics = compute_metrics(y, preds, probs_binary, num_labels=2)

    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")

    return metrics


def save_result(model_name: str, results: dict):
    out_dir = RESULTS_DIR / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved : {out_path}")


def show_transfer_performance(model_name: str, results: dict, base_ds: str):
    in_domain_metrics_path = RESULTS_DIR / base_ds / model_name / "metrics.json"

    with open(in_domain_metrics_path) as f:
        in_domain_metrics = json.load(f)

    in_domain_f1 = in_domain_metrics["test_metrics"]["macro_f1"]
    cross_f1 = results["macro_f1"]
    drop = in_domain_f1 - cross_f1
    drop_pct = (drop / in_domain_f1) * 100 if in_domain_f1 > 0 else 0.0

    print(f"\n {model_name.upper()}:")
    print(f"In-domain: {in_domain_f1:.4f}")
    print(f"Cross-domain: {cross_f1:.4f}")
    print(f"Performance drop: {drop:.4f} ({drop_pct:.1f}%)")


def run_transfer_test(model_name: str):
    print(f"TRANSFER TEST — {model_name.upper()}")

    results = {}

    try:
        results["kaggle2swmh"] = kaggle_to_swmh(model_name)
    except FileNotFoundError as e:
        print(f"{e}")
        results["kaggle2swmh"] = None

    try:
        results["swmh2kaggle"] = swmh_to_kaggle(model_name)
    except FileNotFoundError as e:
        print(f"{e}")
        results["swmh2kaggle"] = None

    save_result(model_name, results)

    if results["kaggle2swmh"] is not None:
        show_transfer_performance(model_name, results["kaggle2swmh"], KAGGLE)

    if results["swmh2kaggle"] is not None:
        show_transfer_performance(model_name, results["swmh2kaggle"], SWMH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY.keys()), required=True
    )
    args = parser.parse_args()

    run_transfer_test(args.model)


if __name__ == "__main__":
    main()
