import sys
import json
import argparse
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, KAGGLE, SWMH

MODEL_ORDER = ["tfidf_lr", "cnn", "bilstm", "distilbert", "mentalbert", "roberta"]

MODEL_LABELS = {
    "tfidf_lr": "TF-IDF + LR",
    "cnn": "CNN",
    "bilstm": "BiLSTM + Attention",
    "distilbert": "DistilBERT",
    "mentalbert": "MentalBERT",
    "roberta": "RoBERTa",
}

METRICS = ["accuracy", "macro_f1", "auc_roc", "mcc", "fnr"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "auc_roc": "AUC-ROC",
    "mcc": "MCC",
    "fnr": "FNR",
}


def load_all_metrics() -> list[dict]:
    records = []

    for dataset in [KAGGLE, SWMH]:
        dataset_dir = RESULTS_DIR / dataset

        if not dataset_dir.exists():
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metrics_path = model_dir / "metrics.json"

            if not metrics_path.exists():
                continue

            try:
                with open(metrics_path) as f:
                    data = json.load(f)

                test_metrics = data.get("test_metrics", {})
                model_name = data["model"]

                record = {
                    "model": model_name,
                    "dataset": dataset,
                    "label": MODEL_LABELS.get(model_name, model_name),
                }

                for metric in METRICS:
                    record[METRIC_LABELS[metric]] = test_metrics.get(metric, None)

                records.append(record)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Could not load {metrics_path}: {e}")
                continue

    return records


def load_transfer_metrics() -> dict:
    results = {}
    transfer_dir = RESULTS_DIR / "transfer"

    if not transfer_dir.exists():
        return results

    for f in transfer_dir.glob("*.json"):
        model_name = f.stem

        try:
            with open(f) as fp:
                data = json.load(fp)

            results[model_name] = {}

            for direction in ["kaggle2swmh", "swmh2kaggle"]:
                val = data.get(direction)

                if isinstance(val, dict):
                    results[model_name][direction] = val

        except Exception as e:
            print(f"  Warning: {f}: {e}")

    return results


def build_comparison_table(records: list[dict], dataset: str) -> pd.DataFrame:
    dataset_records = [r for r in records if r["dataset"] == dataset]

    if not dataset_records:
        return pd.DataFrame()

    df = pd.DataFrame(dataset_records)
    df["sort_key"] = df["model"].apply(
        lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )
    df = df.sort_values("sort_key").drop(columns=["sort_key", "model", "dataset"])
    df = df.set_index("label")

    for metric in df.columns:
        df[metric] = df[metric].apply(lambda x: round(x, 4) if pd.notnull(x) else "—")

    return df


def print_table(df: pd.DataFrame, title: str):
    if df.empty:
        print("No results found.")
        return

    print(f"{title}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string())

    print(f"\n Best per metric:")

    for col in df.columns:
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")

            if col == "FNR":
                best_idx = numeric.idxmin()
                best_val = numeric.min()
            else:
                best_idx = numeric.idxmax()
                best_val = numeric.max()

            print(f"    {col:<12}: {best_idx} ({best_val:.4f})")

        except Exception:
            continue


def plot_comparison(
    records: list[dict], metric: str = "macro_f1", out_path: Path = None
):
    df_all = pd.DataFrame(records)

    if df_all.empty or metric not in df_all.columns:
        print(f"  No data for metric: {metric}")
        return

    df_pivot = df_all.pivot_table(
        index="model", columns="dataset", values=metric, aggfunc="first"
    )

    ordered = [m for m in MODEL_ORDER if m in df_pivot.index]
    df_pivot = df_pivot.loc[ordered]
    df_pivot.index = [MODEL_LABELS.get(m, m) for m in ordered]

    _, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(df_pivot))
    width = 0.35
    colors = ["#2196F3", "#FF5722"]

    for i, (col, color) in enumerate(zip(df_pivot.columns, colors)):
        values = pd.to_numeric(df_pivot[col], errors="coerce")
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=col.upper(),
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if pd.notnull(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df_pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
    ax.set_title(
        f"Model Comparison — {METRIC_LABELS.get(metric, metric)}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out = out_path or (RESULTS_DIR / f"comparison_{metric}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {out}")


def plot_transfer_result(
    in_domain_records: list[dict],
    transfer_data: dict,
    out_path: Path = RESULTS_DIR / "transfer_performance.png",
):

    rows = []

    for model in MODEL_ORDER:
        if model not in transfer_data:
            continue

        label = MODEL_LABELS.get(model, model)
        t = transfer_data[model]

        kaggle_rec = next(
            (
                r
                for r in in_domain_records
                if r["model"] == model and r["dataset"] == KAGGLE
            ),
            None,
        )
        k2s = t.get("kaggle2swmh", {})

        if kaggle_rec and k2s.get("macro_f1") is not None:
            in_f1 = kaggle_rec.get("Macro F1")
            cross_f1 = k2s["macro_f1"]

            if in_f1 and cross_f1:
                rows.append(
                    {
                        "label": label,
                        "direction": "K→S",
                        "drop": round(in_f1 - cross_f1, 4),
                        "in_f1": in_f1,
                        "cross_f1": cross_f1,
                    }
                )

        swmh_rec = next(
            (
                r
                for r in in_domain_records
                if r["model"] == model and r["dataset"] == SWMH
            ),
            None,
        )

        s2k = t.get("swmh2kaggle", {})

        if swmh_rec and s2k.get("macro_f1") is not None:
            in_f1 = swmh_rec.get("Macro F1")
            cross_f1 = s2k["macro_f1"]

            if in_f1 and cross_f1:
                rows.append(
                    {
                        "label": label,
                        "direction": "S→K",
                        "drop": round(in_f1 - cross_f1, 4),
                        "in_f1": in_f1,
                        "cross_f1": cross_f1,
                    }
                )

    if not rows:
        print("  No transfer data — skipping drop plot.")
        return

    df = pd.DataFrame(rows)
    k2s_df = df[df["direction"] == "K→S"].reset_index(drop=True)
    s2k_df = df[df["direction"] == "S→K"].reset_index(drop=True)

    def bar_colors(drops):
        colors = []
        for d in drops:
            if d > 0.10:
                colors.append("#F44336")  # red
            elif d > 0.05:
                colors.append("#FF9800")  # orange
            elif d >= 0:
                colors.append("#4CAF50")  # green
            else:
                colors.append("#2196F3")  # blue — gain

        return colors

    def draw_panel(ax, panel_df, title):
        if panel_df.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            return

        drops = panel_df["drop"].values
        labels = panel_df["label"].values
        colors = bar_colors(drops)
        x = np.arange(len(labels))

        bars = ax.bar(
            x, drops, color=colors, width=0.55, edgecolor="white", linewidth=0.5
        )

        for bar, drop, in_f1, cross_f1 in zip(
            bars, drops, panel_df["in_f1"], panel_df["cross_f1"]
        ):
            y_label = bar.get_height()
            txt = f"{drop:+.4f}\n({in_f1:.3f}→{cross_f1:.3f})"
            va = "bottom" if y_label >= 0 else "top"
            offset = 0.003 if y_label >= 0 else -0.003
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_label + offset,
                txt,
                ha="center",
                va=va,
                fontsize=7,
            )

        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(0.10, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Macro F1 Drop (positive = worse)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="#F44336", label="Large drop  (>0.10)"),
                Patch(facecolor="#FF9800", label="Moderate    (0.05–0.10)"),
                Patch(facecolor="#4CAF50", label="Small drop  (<0.05)"),
                Patch(facecolor="#2196F3", label="Gain        (negative)"),
            ],
            fontsize=8,
            loc="upper right",
        )

    n_cols = max(len(k2s_df), len(s2k_df), 1)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, n_cols * 1.2 + 4), 8))

    draw_panel(axes[0], k2s_df, "K→S: Kaggle-trained → SWMH tested")
    draw_panel(axes[1], s2k_df, "S→K: SWMH-trained → Kaggle tested")

    fig.suptitle(
        "Cross-Dataset Generalisation — Performance Drop / Gain (Macro F1)\n"
        "Positive = performance dropped | Negative = performance improved",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved : {out_path.name}")


def main():
    records = load_all_metrics()

    if not records:
        print("No metrics.json files found in results/ directory.")
        return

    print(f"Found {len(records)} model-dataset combinations.")

    kaggle_df = build_comparison_table(records, KAGGLE)
    print_table(kaggle_df, "KAGGLE SUICIDEWATCH — Binary Classification")

    swmh_df = build_comparison_table(records, SWMH)
    print_table(swmh_df, "SWMH — Multi-Class Classification (5 Classes)")

    kaggle_path = RESULTS_DIR / "master_results_kaggle.csv"
    swmh_path = RESULTS_DIR / "master_results_swmh.csv"

    if not kaggle_df.empty:
        kaggle_df.to_csv(kaggle_path)
        print(f"\n  Kaggle table saved → {kaggle_path}")

    if not swmh_df.empty:
        swmh_df.to_csv(swmh_path)
        print(f"  SWMH table saved   → {swmh_path}")

    for metric in METRICS:
        plot_comparison(records, METRIC_LABELS[metric])

    transfer_data = load_transfer_metrics()
    print(transfer_data)
    plot_transfer_result(records, transfer_data)


if __name__ == "__main__":
    main()
