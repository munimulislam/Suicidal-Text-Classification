import sys
import re
import html
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import torch
import argparse
import random
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SEED,
    KAGGLE,
    SWMH,
    KAGGLE_CSV,
    SWMH_TRAIN_CSV,
    SWMH_TEST_CSV,
    SWMH_VAL_CSV,
    PROCESSED_DIR,
    KAGGLE_LABEL_MAP,
    SWMH_LABEL_MAP,
)

URL_RE = re.compile(r"http\S+|www\.\S+|https\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REDDIT_RE = re.compile(r"u/\w+|r/\w+")
ARTEFACT_RE = re.compile(r"\[deleted\]|\[removed\]", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<.*?>")
EMAIL_RE = re.compile(r"\S+@\S+")
MULTISPACE = re.compile(r"\s+")
NON_PRINT = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
REPEAT_CHR = re.compile(r"(.)\1{2,}")
PLACEHOLDER_RE = re.compile(r"<(url|email|user)>", re.IGNORECASE)

_STOPWORDS = None
_LEMMATIZER = None

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _init_nltk_lazy():
    global _STOPWORDS, _LEMMATIZER

    if _STOPWORDS is None or _LEMMATIZER is None:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"):
            try:
                nltk.data.find(
                    f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
                )
            except LookupError:
                nltk.download(pkg, quiet=True)

        _STOPWORDS = set(stopwords.words("english"))
        _LEMMATIZER = WordNetLemmatizer()

    return _STOPWORDS, _LEMMATIZER


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = html.unescape(text)
    t = ARTEFACT_RE.sub("", t)
    t = HTML_TAG_RE.sub("", t)
    t = EMAIL_RE.sub(" <email> ", t)
    t = URL_RE.sub(" <url> ", t)
    t = REDDIT_RE.sub(" <user> ", t)
    t = MENTION_RE.sub(" <user> ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)
    t = NON_PRINT.sub(" ", t)
    t = REPEAT_CHR.sub(r"\1\1", t)
    t = MULTISPACE.sub(" ", t).strip()

    return t


def lemmatize_txt(text: str) -> str:
    if not isinstance(text, str):
        return ""

    stop, lem = _init_nltk_lazy()

    if stop is None or lem is None:
        return text

    from nltk.tokenize import word_tokenize

    text = PLACEHOLDER_RE.sub("", text)
    tokens = word_tokenize(text.lower())
    tokens = [
        lem.lemmatize(t) for t in tokens if t.isalpha() and t not in stop and len(t) > 2
    ]

    return " ".join(tokens)


def drop_empty_and_short(
    df: pd.DataFrame, text_col: str = "text", min_words: int = 3
) -> pd.DataFrame:

    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)
    word_counts = df[text_col].str.split().str.len()
    df = df[word_counts >= min_words].reset_index(drop=True)

    return df


def drop_duplicates_clean(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> pd.DataFrame:

    df = df.drop_duplicates(subset=[text_col, label_col])
    conflict = df.groupby(text_col)[label_col].nunique()
    bad_texts = set(conflict[conflict > 1].index)

    if bad_texts:
        print(f"Removing {len(bad_texts)} texts with conflicting labels")

    df = df[~df[text_col].isin(bad_texts)].reset_index(drop=True)

    return df


def remove_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, text_col: str = "text"
) -> tuple:

    train_texts = set(train[text_col])
    val_texts = set(val[text_col])

    val_before = len(val)
    test_before = len(test)

    val = val[~val[text_col].isin(train_texts)].reset_index(drop=True)
    test = test[~test[text_col].isin(train_texts)].reset_index(drop=True)
    test = test[~test[text_col].isin(val_texts)].reset_index(drop=True)

    print(
        f"Leakage removed — val: {val_before}→{len(val)} "
        f"test: {test_before}→{len(test)}"
    )

    return train, val, test


def save_dataset(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    weights_tensor,
) -> None:

    out = PROCESSED_DIR / "new" / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    full = pd.concat([train, val, test], ignore_index=True)
    full.to_csv(out / "cleaned.csv", index=False)

    np.save(str(out / "class_weights.npy"), weights_tensor.numpy())

    print(f"Saved at: {out}")
    print(f"train={len(train)} | val={len(val)} | test={len(test)}")


def print_split_distribution(train, val, test):
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split["label"].value_counts(normalize=True).sort_index()
        dist_str = " | ".join([f"class {k}: {v:.1%}" for k, v in dist.items()])
        print(f"{name:5s}: {dist_str}")


def sanity_check(df: pd.DataFrame, dataset_name: str, n: int = 3):
    print(f"\n{'='*60}")
    print(f"Sanity Check — {dataset_name} ({n} random samples)")
    print(f"{'='*60}")

    samples = df.sample(n, random_state=SEED)

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n  Sample {i} | Label: {row['label']}")
        print(f"ORIGINAL : {str(row['text'])[:120]}")
        print(f"CLEANED : {str(row['text_clean'])[:120]}")
        print(f"CLEANED + LEMMATIZED : {str(row['text_lemmatized'])[:120]}")


def get_class_weights(labels, device=None):
    classes = np.unique(labels)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)

    weights_dict = dict(zip(classes.tolist(), weights.tolist()))
    weights_tensor = torch.tensor(
        [weights_dict[i] for i in range(len(classes))], dtype=torch.float
    )

    if device is not None:
        weights_tensor = weights_tensor.to(device)

    print(f"Class weights: {weights_dict}")

    return weights_dict, weights_tensor


def process_kaggle():
    from sklearn.model_selection import train_test_split

    try:
        df = pd.read_csv(KAGGLE_CSV)
    except FileNotFoundError:
        print(f"ERROR: File not found at {KAGGLE_CSV}")
        print("Place suicide_watch.csv in data/kaggle/")
        return

    print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")

    df = df.rename(columns={"class": "label_str"})
    df = df[["text", "label_str"]]

    df["label"] = df["label_str"].map(KAGGLE_LABEL_MAP)
    unmapped = df["label"].isna().sum()

    if unmapped > 0:
        print(f"  WARNING: {unmapped} rows had unmapped labels — dropping them")
        df = df.dropna(subset=["label"])

    print(f"Label distribution:\n{df['label'].value_counts()}")

    before = len(df)
    df = drop_empty_and_short(df)
    df = drop_duplicates_clean(df)
    print(
        f"Removed empty & duplicates: {before - len(df):,} rows - {len(df):,} remaining"
    )

    df["text_clean"] = df["text"].progress_apply(clean_text)
    df["text_lemmatized"] = df["text_clean"].progress_apply(lemmatize_txt)

    count = df["text_lemmatized"].str.split().str.len()
    before = len(df)
    df = df[count >= 5].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with <5 tokens after cleaning")

    df["label"] = df["label"].astype(int)

    train, test = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=SEED
    )
    train, val = train_test_split(
        train, test_size=0.10 / 0.90, stratify=train["label"], random_state=SEED
    )
    train, val, test = remove_leakage(train, val, test)
    print_split_distribution(train, val, test)

    _, weights_tensor = get_class_weights(train["label"].values)

    save_dataset(train, val, test, KAGGLE, weights_tensor)
    sanity_check(df, KAGGLE)


def process_swmh():
    try:
        train = pd.read_csv(SWMH_TRAIN_CSV)
        val = pd.read_csv(SWMH_VAL_CSV)
        test = pd.read_csv(SWMH_TEST_CSV)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Place SWMH train/val/test CSVs in data/swmh/")
        return

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    for d in (train, val, test):
        d.columns = [c.strip().lower() for c in d.columns]

    print(f"Train columns: {list(train.columns)}")
    print(f"Unique labels in train: {train['label'].unique()}")

    for name, split in [(TRAIN, train), (VAL, val), (TEST, test)]:
        split["label_str"] = split["label"].copy()
        split["label"] = split["label"].map(SWMH_LABEL_MAP)

        unmapped = split["label"].isna().sum()

        if unmapped > 0:
            print(f"WARNING: {name} has {unmapped} unmapped labels")
            print(f"Unrecognised: {split[split['label'].isna()]['label_str'].unique()}")
            split = split.dropna(subset=["label"])

        print(f"Label distribution:\n{split['label'].value_counts()}")

        before = len(split)
        split = drop_empty_and_short(split)
        split = drop_duplicates_clean(split)
        print(f"  {name}: {before:,} → {len(split):,}")

        split["label"] = split["label"].astype(int)

        if name == TRAIN:
            train = split
        elif name == VAL:
            val = split
        else:
            test = split

    train, val, test = remove_leakage(train, val, test)

    for name, split in [(TRAIN, train), (VAL, val), (TEST, test)]:
        print(f"Cleaning {name}...")
        split["text_clean"] = split["text"].progress_apply(clean_text)
        split["text_lemmatized"] = split["text_clean"].progress_apply(lemmatize_txt)

        count = split["text_lemmatized"].str.split().str.len()
        before = len(split)
        split = split[count >= 5].reset_index(drop=True)

        print(f"{name}: dropped {before - len(split)} short posts")

        if name == TRAIN:
            train = split
        elif name == VAL:
            val = split
        else:
            test = split

    train["label"] = train["label"].astype(int)
    val["label"] = val["label"].astype(int)
    test["label"] = test["label"].astype(int)

    print_split_distribution(train, val, test)

    _, weights_tensor = get_class_weights(train["label"].values)
    save_dataset(train, val, test, SWMH, weights_tensor)
    sanity_check(train, SWMH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for Mental Health Crisis Detection"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[KAGGLE, SWMH],
        default=None,
        help="Dataset to process. If not provided, processes both.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == KAGGLE:
        print("Processing Kaggle dataset...")
        process_kaggle()
    elif args.dataset == SWMH:
        print("Processing SWMH dataset...")
        process_swmh()
    else:
        print("No dataset specified — processing both...")
        process_kaggle()
        process_swmh()

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
