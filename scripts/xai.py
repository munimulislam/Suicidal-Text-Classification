import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
from lime.lime_text import LimeTextExplainer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    SEED,
    KAGGLE,
    SWMH,
    KAGGLE_PROCESSED,
    SWMH_PROCESSED,
    KAGGLE_LABELS,
    SWMH_LABELS,
    CHECKPOINT_OUT_DIR,
    XAI_DIR,
    MAX_LEN,
)

from utils.transformer_utils import MODEL_REGISTRY

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

TRANSFORMER_FROM_MODEL = {
    "bert": "bert",
    "distilbert": "distilbert",
    "mentalbert": "mentalbert",
    "roberta": "roberta",
    "bert_cnn": "bert",
    "bert_bilstm": "bert",
    "mentalbert_cnn": "mentalbert",
    "roberta_cnn": "roberta",
    "roberta_bilstm": "roberta",
    "mentalbert_cnn_bilstm": "mentalbert",
}

CONFIDENCE_THRESHOLD = 0.7


def load_model_and_tokenizer(model_name: str, dataset_name: str, num_labels: int):
    ckpt_dir = CHECKPOINT_IN_DIR / f"{model_name}_{dataset_name}"
    ckpt_path = ckpt_dir / "best_model.pt"
    transformer = TRANSFORMER_FROM_MODEL.get(model_name, "roberta")
    hf_model_id = MODEL_REGISTRY[transformer]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Ensure training completed for {model_name} on {dataset_name}"
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, hf_model_id


def make_predict_fn(model, tokenizer):
    """
    Returns a prediction function for SHAP and LIME.

    Uses lru_cache on tokenisation to avoid re-encoding
    the same text repeatedly during perturbation-based XAI.
    """

    # Tuple-hashable wrapper for lru_cache
    @lru_cache(maxsize=10000)
    def _cached_encode(text: str):
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0))

    def predict_fn(texts: list) -> np.ndarray:
        model.eval()
        all_probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            input_ids_list = []
            mask_list = []
            for text in batch:
                ids, mask = _cached_encode(str(text))
                input_ids_list.append(ids)
                mask_list.append(mask)

            input_ids = torch.stack(input_ids_list).to(device)
            attention_mask = torch.stack(mask_list).to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

        return np.array(all_probs)

    return predict_fn


def get_sample_sets(
    predict_fn, texts: list, true_labels: list, label_names: list, n_per_class: int = 3
):

    print(f"Computing predictions on {len(texts)} test samples...")
    all_probs = predict_fn(texts)
    pred_labels = np.argmax(all_probs, axis=1)
    max_probs = np.max(all_probs, axis=1)

    correct_samples = {}
    false_positive_cases = []
    false_negative_cases = []

    for class_idx, class_name in enumerate(label_names):
        correct_samples[class_name] = []

    for i, (text, true_label, pred_label, confidence) in enumerate(
        zip(texts, true_labels, pred_labels, max_probs)
    ):
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Correct predictions
        if true_label == pred_label:
            class_name = label_names[true_label]
            if len(correct_samples[class_name]) < n_per_class:
                correct_samples[class_name].append(
                    {
                        "idx": i,
                        "text": text,
                        "true_label": label_names[true_label],
                        "pred_label": label_names[pred_label],
                        "confidence": float(confidence),
                        "probs": all_probs[i].tolist(),
                    }
                )

        crisis_label = 1 if len(label_names) == 2 else 0

        if true_label == crisis_label and pred_label != crisis_label:
            if len(false_negative_cases) < n_per_class:
                false_negative_cases.append(
                    {
                        "idx": i,
                        "text": text,
                        "true_label": label_names[true_label],
                        "pred_label": label_names[pred_label],
                        "confidence": float(confidence),
                        "probs": all_probs[i].tolist(),
                    }
                )

        if pred_label == crisis_label and true_label != crisis_label:
            if len(false_positive_cases) < n_per_class:
                false_positive_cases.append(
                    {
                        "idx": i,
                        "text": text,
                        "true_label": label_names[true_label],
                        "pred_label": label_names[pred_label],
                        "confidence": float(confidence),
                        "probs": all_probs[i].tolist(),
                    }
                )

    total_correct = sum(len(v) for v in correct_samples.values())
    print(f"Confident correct: {total_correct}")
    print(f"False negatives (missed crises): {len(false_negative_cases)}")
    print(f"False positives (false alarms): {len(false_positive_cases)}")

    return correct_samples, false_negative_cases, false_positive_cases


def run_shap(
    predict_fn,
    tokenizer,
    correct_samples: dict,
    fn_cases: list,
    fp_cases: list,
    label_names: list,
    out_dir: Path,
):

    print(f"\nRunning SHAP with Text masker...")
    out_dir.mkdir(parents=True, exist_ok=True)

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=label_names)

    shap_token_importance = {name: Counter() for name in label_names}

    def explain_sample(
        text: str,
        true_label: str,
        pred_label: str,
        sample_type: str,
        filename_prefix: str,
    ):
        try:
            sv = explainer([text])

            token_strings = sv.data[0]
            shap_matrix = sv.values[0]

            # shap.plots.text renders HTML with colour-coded tokens
            html_content = shap.plots.text(sv, display=False)
            if html_content:
                with open(out_dir / f"{filename_prefix}_shap.html", "w") as f:
                    f.write(str(html_content))

            primary_class = 1 if len(label_names) == 2 else 0
            token_shap = shap_matrix[:, primary_class]

            # Sort by absolute importance
            top_n = min(20, len(token_strings))
            top_idx = np.argsort(np.abs(token_shap))[-top_n:][::-1]

            top_tokens = [token_strings[j] for j in top_idx]
            top_values = [token_shap[j] for j in top_idx]

            colors = ["#d73027" if v > 0 else "#4575b4" for v in top_values]

            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), top_values, color=colors)
            plt.yticks(range(top_n), top_tokens, fontsize=9)
            plt.xlabel("SHAP Value (red=pushes toward crisis, blue=away)")
            plt.title(
                f"SHAP — {sample_type}\n"
                f"True: {true_label} | Pred: {pred_label}\n"
                f"Text: {str(text)[:60]}..."
            )
            plt.axvline(x=0, color="black", linewidth=0.8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{filename_prefix}_shap_bar.png", dpi=300)
            plt.close()

            # Accumulate token importance
            for tok, val in zip(token_strings, token_shap):
                tok_clean = str(tok).replace("Ġ", "").replace("##", "").strip()
                if tok_clean and tok_clean not in [
                    "[CLS]",
                    "[SEP]",
                    "[PAD]",
                    "<s>",
                    "</s>",
                ]:
                    shap_token_importance[true_label][tok_clean] += abs(float(val))

            print(f"    Explained: {filename_prefix}", flush=True)

        except Exception as e:
            print(f"    Warning: SHAP failed for {filename_prefix}: {e}")

    # Correct predictions
    for class_name, samples in correct_samples.items():
        for i, s in enumerate(samples):
            explain_sample(
                text=s["text"],
                true_label=s["true_label"],
                pred_label=s["pred_label"],
                sample_type=f"Correct [{class_name}]",
                filename_prefix=f"correct_{class_name}_{i+1}",
            )

    # False negatives
    for i, s in enumerate(fn_cases):
        explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            pred_label=s["pred_label"],
            sample_type="FALSE NEGATIVE (crisis missed)",
            filename_prefix=f"false_negative_{i+1}",
        )

    # False positives
    for i, s in enumerate(fp_cases):
        explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            pred_label=s["pred_label"],
            sample_type="FALSE POSITIVE (false alarm)",
            filename_prefix=f"false_positive_{i+1}",
        )

    shap_summary = {
        class_name: [
            {"token": tok, "total_importance": score}
            for tok, score in counter.most_common(20)
        ]
        for class_name, counter in shap_token_importance.items()
    }

    with open(out_dir / "shap_token_importance.json", "w") as f:
        json.dump(shap_summary, f, indent=2)

    print(f"SHAP complete → {out_dir}")

    return shap_token_importance


def run_lime(
    predict_fn,
    correct_samples: dict,
    fn_cases: list,
    fp_cases: list,
    label_names: list,
    out_dir: Path,
):

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        import subprocess

        subprocess.run(["pip", "install", "lime", "-q"])
        from lime.lime_text import LimeTextExplainer

    print(f"\n  Running LIME (num_samples=1000 for stability)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = LimeTextExplainer(class_names=label_names, random_state=SEED)

    lime_token_importance = {name: Counter() for name in label_names}

    def explain_sample(
        text: str,
        true_label: str,
        pred_label: str,
        label_idx: int,
        sample_type: str,
        filename_prefix: str,
    ):

        try:
            exp = explainer.explain_instance(
                str(text),
                predict_fn,
                num_features=20,
                num_samples=1000,
                labels=[label_idx],
            )

            # HTML explanation
            with open(out_dir / f"{filename_prefix}_lime.html", "w") as f:
                f.write(exp.as_html())

            # Bar plot with real words
            word_weights = exp.as_list(label=label_idx)
            if not word_weights:
                return

            words = [w[0] for w in word_weights]
            weights = [w[1] for w in word_weights]
            colors = ["#d73027" if w > 0 else "#4575b4" for w in weights]

            plt.figure(figsize=(10, 5))
            plt.barh(range(len(words)), weights, color=colors)
            plt.yticks(range(len(words)), words, fontsize=9)
            plt.xlabel("LIME Weight")
            plt.title(
                f"LIME — {sample_type}\n"
                f"True: {true_label} | Pred: {pred_label}\n"
                f"Text: {str(text)[:60]}..."
            )
            plt.axvline(x=0, color="black", linewidth=0.8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{filename_prefix}_lime_bar.png", dpi=300)
            plt.close()

            # Accumulate
            for word, weight in word_weights:
                word_clean = word.strip().lower()
                if word_clean:
                    lime_token_importance[true_label][word_clean] += abs(float(weight))

            print(f"    Explained: {filename_prefix}", flush=True)

        except Exception as e:
            print(f"    Warning: LIME failed for {filename_prefix}: {e}")

    # Correct predictions
    for class_name, samples in correct_samples.items():
        label_idx = label_names.index(class_name)
        for i, s in enumerate(samples):
            explain_sample(
                text=s["text"],
                true_label=s["true_label"],
                pred_label=s["pred_label"],
                label_idx=label_idx,
                sample_type=f"Correct [{class_name}]",
                filename_prefix=f"correct_{class_name}_{i+1}",
            )

    # False negatives
    crisis_label = label_names[1] if len(label_names) == 2 else label_names[0]
    crisis_idx = 1 if len(label_names) == 2 else 0

    for i, s in enumerate(fn_cases):
        explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            pred_label=s["pred_label"],
            label_idx=crisis_idx,
            sample_type="FALSE NEGATIVE (crisis missed)",
            filename_prefix=f"false_negative_{i+1}",
        )

    # False positives
    for i, s in enumerate(fp_cases):
        explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            pred_label=s["pred_label"],
            label_idx=crisis_idx,
            sample_type="FALSE POSITIVE (false alarm)",
            filename_prefix=f"false_positive_{i+1}",
        )

    lime_summary = {
        class_name: [
            {"token": tok, "total_importance": score}
            for tok, score in counter.most_common(20)
        ]
        for class_name, counter in lime_token_importance.items()
    }

    with open(out_dir / "lime_token_importance.json", "w") as f:
        json.dump(lime_summary, f, indent=2)

    print(f"LIME complete → {out_dir}")

    return lime_token_importance


def run_attention(
    model,
    tokenizer,
    correct_samples: dict,
    fn_cases: list,
    label_names: list,
    out_dir: Path,
):

    print(f"\nRunning Attention Visualisation...")
    out_dir.mkdir(parents=True, exist_ok=True)

    attention_token_importance = {name: Counter() for name in label_names}

    if hasattr(model, "bert"):
        encoder = model.bert
    elif hasattr(model, "roberta"):
        encoder = model.roberta
    elif hasattr(model, "distilbert"):
        encoder = model.distilbert
    elif hasattr(model, "base_model"):
        encoder = model.base_model
    else:
        raise ValueError(
            f"Cannot identify transformer encoder in model. "
            f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    def explain_sample(
        text: str,
        true_label: str,
        pred_label: str,
        sample_type: str,
        filename_prefix: str,
    ):

        try:
            enc = tokenizer(
                str(text),
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            # Get tokens — remove padding
            seq_len = attention_mask[0].sum().item()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())[
                :seq_len
            ]

            # Last layer, average across heads, CLS token attention
            last_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
            avg_attn = last_attn[0].mean(dim=0)  # (seq, seq)
            cls_attn = avg_attn[0, :seq_len].cpu().numpy()  # (seq_len,)

            # Normalise
            cls_attn_norm = (cls_attn - cls_attn.min()) / (
                cls_attn.max() - cls_attn.min() + 1e-8
            )

            # Show at most 30 tokens for readability
            n_show = min(seq_len, 30)
            display_tokens = tokens[:n_show]
            display_attn = cls_attn_norm[:n_show]

            fig, ax = plt.subplots(figsize=(max(12, n_show * 0.5), 2.5))
            im = ax.imshow(
                display_attn.reshape(1, -1), cmap="Blues", aspect="auto", vmin=0, vmax=1
            )
            ax.set_xticks(range(n_show))
            ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.02)
            ax.set_title(
                f"Attention — {sample_type}\n"
                f"True: {true_label} | Pred: {pred_label}\n"
                f"⚠ Note: Attention shows model focus, not causal importance",
                fontsize=9,
            )
            plt.tight_layout()
            plt.savefig(
                out_dir / f"{filename_prefix}_attention.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Accumulate top tokens
            top_idx = np.argsort(cls_attn[:seq_len])[-10:][::-1]
            for j in top_idx:
                tok = tokens[j].replace("Ġ", "").replace("##", "").strip().lower()
                if tok and tok not in ["[cls]", "[sep]", "[pad]", "<s>", "</s>"]:
                    attention_token_importance[true_label][tok] += float(cls_attn[j])

            print(f"    Explained: {filename_prefix}", flush=True)

        except Exception as e:
            print(f"    Warning: Attention failed for {filename_prefix}: {e}")

    # Correct predictions
    for class_name, samples in correct_samples.items():
        for i, s in enumerate(samples):
            explain_sample(
                text=s["text"],
                true_label=s["true_label"],
                pred_label=s["pred_label"],
                sample_type=f"Correct [{class_name}]",
                filename_prefix=f"correct_{class_name}_{i+1}",
            )

    # False negatives
    for i, s in enumerate(fn_cases):
        explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            pred_label=s["pred_label"],
            sample_type="FALSE NEGATIVE (crisis missed)",
            filename_prefix=f"false_negative_{i+1}",
        )

    # Save summary
    attn_summary = {
        class_name: [
            {"token": tok, "total_attention": score}
            for tok, score in counter.most_common(20)
        ]
        for class_name, counter in attention_token_importance.items()
    }
    with open(out_dir / "attention_token_importance.json", "w") as f:
        json.dump(attn_summary, f, indent=2)

    print(f"Attention complete → {out_dir}")
    return attention_token_importance


def generate_clinical_insights(
    shap_importance: dict,
    lime_importance: dict,
    attn_importance: dict,
    label_names: list,
    out_path: Path,
):

    lines = []
    lines.append("")

    all_clinical_tokens = {}

    for class_name in label_names:
        shap_counter = shap_importance.get(class_name, Counter())
        lime_counter = lime_importance.get(class_name, Counter())
        attn_counter = attn_importance.get(class_name, Counter())

        # Normalise each method's scores to [0,1]
        def normalise_counter(c: Counter) -> dict:
            if not c:
                return {}
            max_val = max(c.values()) if c else 1.0
            return {k: v / max_val for k, v in c.items()}

        shap_norm = normalise_counter(shap_counter)
        lime_norm = normalise_counter(lime_counter)
        attn_norm = normalise_counter(attn_counter)

        # Union of all tokens
        all_tokens = set(shap_norm) | set(lime_norm) | set(attn_norm)

        # Combined score — frequency weighting across methods
        # Tokens present in all 3 get a score proportional to all three
        combined = {}
        for tok in all_tokens:
            s = shap_norm.get(tok, 0)
            l = lime_norm.get(tok, 0)
            a = attn_norm.get(tok, 0)
            n_methods = sum([s > 0, l > 0, a > 0])  # how many methods agree
            if n_methods >= 2:  # at least 2 methods must agree
                combined[tok] = (s + l + a) * n_methods  # reward agreement

        top_tokens = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:15]

        # Tokens only in SHAP+LIME (most reliable pair without attention)
        shap_lime_only = set(shap_norm) & set(lime_norm) - set(attn_norm)
        all_three = set(shap_norm) & set(lime_norm) & set(attn_norm)

        lines.append(f"[{class_name.upper()}]")
        lines.append(
            f"  All 3 methods agree: {', '.join(sorted(all_three)[:10]) or 'None'}"
        )
        lines.append(
            f"  SHAP + LIME agree:   {', '.join(sorted(shap_lime_only)[:10]) or 'None'}"
        )
        lines.append(f"  Top 15 by combined score:")
        for rank, (tok, score) in enumerate(top_tokens, 1):
            methods = []
            if tok in shap_norm:
                methods.append("SHAP")
            if tok in lime_norm:
                methods.append("LIME")
            if tok in attn_norm:
                methods.append("ATTN")
            lines.append(
                f"    {rank:2d}. {tok:<20} score={score:.3f} [{', '.join(methods)}]"
            )
        lines.append("")

        all_clinical_tokens[class_name] = top_tokens

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    # Also save as JSON for downstream use
    json_out = out_path.parent / "clinical_insights.json"
    with open(json_out, "w") as f:
        json.dump(
            {
                cls: [{"token": t, "score": s} for t, s in tokens]
                for cls, tokens in all_clinical_tokens.items()
            },
            f,
            indent=2,
        )

    print(f"\n  Clinical insights saved → {out_path}")
    for line in lines[:30]:  # print first 30 lines
        print(f"  {line}")

    return all_clinical_tokens


def run_xai(model_name: str, dataset_name: str):
    print(f"XAI ANALYSIS — {model_name.upper()} on {dataset_name.upper()}")

    processed_dir = KAGGLE_PROCESSED if dataset_name == KAGGLE else SWMH_PROCESSED
    label_names = KAGGLE_LABELS if dataset_name == KAGGLE else SWMH_LABELS
    num_labels = len(label_names)

    xai_out = XAI_DIR / dataset_name / model_name
    xai_out.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(processed_dir / "test.csv")
    texts = test_df["text_clean"].fillna("").astype(str).tolist()
    labels = test_df["label"].tolist()
    print(f" Test set: {len(texts):,} samples")

    model, tokenizer, hf_model_id = load_model_and_tokenizer(
        model_name, dataset_name, num_labels
    )

    predict_fn = make_predict_fn(model, tokenizer)

    # Sanity check
    sample_probs = predict_fn(texts[:3])

    for i, probs in enumerate(sample_probs):
        pred = label_names[np.argmax(probs)]
        print(f"[{i+1}] {pred} ({max(probs):.3f})")

    correct_samples, fn_cases, fp_cases = get_sample_sets(
        predict_fn, texts, labels, label_names, n_per_class=3
    )

    # Save misclassification cases to JSON for report
    misclass_path = xai_out / "misclassification_cases.json"

    with open(misclass_path, "w") as f:
        json.dump(
            {
                "false_negatives_missed_crises": fn_cases,
                "false_positives_false_alarms": fp_cases,
            },
            f,
            indent=2,
        )

    print(f"Misclassification cases saved → {misclass_path}")

    # LIME
    lime_importance = run_lime(
        predict_fn=predict_fn,
        correct_samples=correct_samples,
        fn_cases=fn_cases,
        fp_cases=fp_cases,
        label_names=label_names,
        out_dir=xai_out / "lime",
    )

    # Attention
    attn_importance = run_attention(
        model=model,
        tokenizer=tokenizer,
        correct_samples=correct_samples,
        fn_cases=fn_cases,
        label_names=label_names,
        out_dir=xai_out / "attention",
    )

    # SHAP
    try:
        shap_importance = run_shap(
            predict_fn=predict_fn,
            tokenizer=tokenizer,
            correct_samples=correct_samples,
            fn_cases=fn_cases,
            fp_cases=fp_cases,
            label_names=label_names,
            out_dir=xai_out / "shap",
        )
    except Exception as e:
        print(f"  SHAP failed: {e}")
        print(f"  Continuing with LIME + Attention only")
        shap_importance = {name: Counter() for name in label_names}

    generate_clinical_insights(
        shap_importance=shap_importance,
        lime_importance=lime_importance,
        attn_importance=attn_importance,
        label_names=label_names,
        out_path=xai_out / "clinical_insights.txt",
    )

    print(f"\nXAI complete — {model_name} on {dataset_name}")
    print(f"All outputs → {xai_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI — SHAP (Text masker) + LIME + Attention"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=[KAGGLE, SWMH], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else [KAGGLE, SWMH]
    for d in datasets:
        run_xai(args.model, d)
