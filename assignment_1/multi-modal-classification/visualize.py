"""
Visualization utilities for experiment results.
Uses seaborn for polished, publication-quality charts.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from config import CATEGORIES, PLOTS_DIR

# Use non-interactive backend for saving plots
matplotlib.use("Agg")

# ─── Seaborn theme ───────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.3,
    rc={
        "figure.figsize": (10, 7),
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    },
)

PALETTE = sns.color_palette("husl", 10)
MODEL_PALETTE = {"ViT-B-32": "#4C72B0", "ViT-B-16": "#DD8452"}


def _save_fig(fig, filename):
    """Save figure and log."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {filepath}")
    return filepath


def plot_confusion_matrix(
    cm, title="Confusion Matrix", filename="confusion_matrix.png"
):
    """Plot and save a confusion matrix heatmap using seaborn."""
    short_names = [c.replace(" and ", " & ").title()[:18] for c in CATEGORIES]

    num_classes = len(CATEGORIES)
    is_large = num_classes > 30

    # Normalize for display (percentages), keep raw for annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(20, 18) if is_large else (12, 10))

    if not is_large:
        # Annotate with both count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"
    else:
        annot = False

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        linewidths=0 if is_large else 0.5,
        linecolor="white",
        cbar_kws={"label": "Proportion", "shrink": 0.8},
        square=True,
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=15)

    if is_large:
        plt.xticks(rotation=90, ha="center", fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
    else:
        plt.xticks(rotation=40, ha="right")
        plt.yticks(rotation=0)

    return _save_fig(fig, filename)


def plot_zero_shot_comparison(results_by_model, filename="zero_shot_comparison.png"):
    """Grouped bar chart comparing zero-shot accuracy across models."""
    records = []
    for model, metrics in results_by_model.items():
        records.append(
            {"Model": model, "Metric": "Accuracy", "Score": metrics["accuracy"]}
        )
        records.append(
            {"Model": model, "Metric": "F1 (weighted)", "Score": metrics["f1_weighted"]}
        )
        records.append(
            {"Model": model, "Metric": "F1 (macro)", "Score": metrics["f1_macro"]}
        )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Model",
        y="Score",
        hue="Metric",
        palette="deep",
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
    )

    # Value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10, padding=3)

    ax.set_ylim(0, min(1.05, df["Score"].max() + 0.08))
    ax.set_title("Zero-Shot Classification: Model Comparison", pad=15)
    ax.set_ylabel("Score")
    ax.legend(title="Metric", loc="lower right")
    sns.despine(left=True)

    return _save_fig(fig, filename)


def plot_few_shot_curves(results_by_model, filename="few_shot_curves.png"):
    """Line plot: accuracy vs number of shots, one line per model."""
    records = []
    for model_name, k_results in results_by_model.items():
        for k, res in k_results.items():
            records.append(
                {
                    "Shots (k)": k,
                    "Accuracy": res["accuracy_mean"],
                    "std": res["accuracy_std"],
                    "Model": model_name,
                }
            )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Shots (k)",
        y="Accuracy",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette="deep",
        ax=ax,
        linewidth=2.5,
        markersize=10,
    )

    # Error bars
    for model_name in df["Model"].unique():
        sub = df[df["Model"] == model_name]
        ax.fill_between(
            sub["Shots (k)"],
            sub["Accuracy"] - sub["std"],
            sub["Accuracy"] + sub["std"],
            alpha=0.15,
        )

    ax.set_ylim(0, 1.05)
    ax.set_title("Few-Shot Classification: Accuracy vs Number of Shots", pad=15)
    ax.set_xlabel("Number of Shots per Class (k)")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Model")
    sns.despine(left=True)

    return _save_fig(fig, filename)


def plot_zero_vs_few_shot(zero_results, few_results, filename="zero_vs_few_shot.png"):
    """Bar chart comparing zero-shot vs best few-shot accuracy for each model."""
    records = []
    for model in zero_results:
        records.append(
            {
                "Model": model,
                "Method": "Zero-Shot",
                "Accuracy": zero_results[model]["accuracy"],
            }
        )
        if model in few_results and few_results[model]:
            best_k = max(
                few_results[model].keys(),
                key=lambda k: few_results[model][k]["accuracy_mean"],
            )
            records.append(
                {
                    "Model": model,
                    "Method": f"Few-Shot (k={best_k})",
                    "Accuracy": few_results[model][best_k]["accuracy_mean"],
                }
            )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Model",
        y="Accuracy",
        hue="Method",
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=11, padding=3)

    ax.set_ylim(0, min(1.05, df["Accuracy"].max() + 0.08))
    ax.set_title("Zero-Shot vs Few-Shot Classification", pad=15)
    ax.set_ylabel("Accuracy")
    ax.legend(title="Method")
    sns.despine(left=True)

    return _save_fig(fig, filename)


def plot_label_distribution(label_dist, filename="label_distribution.png"):
    """Horizontal bar chart of dataset label distribution."""
    cats = [
        CATEGORIES[i].replace(" and ", " & ").title()[:20]
        for i in sorted(label_dist.keys())
    ]
    counts = [label_dist[i] for i in sorted(label_dist.keys())]

    df = pd.DataFrame({"Category": cats, "Count": counts})
    df = df.sort_values("Count", ascending=True)

    is_large = len(cats) > 30
    fig, ax = plt.subplots(figsize=(10, min(max(6, len(cats) * 0.25), 30)))

    sns.barplot(
        data=df,
        x="Count",
        y="Category",
        hue="Category",
        legend=False,
        palette="viridis",
        ax=ax,
        edgecolor="white",
    )
    for i, (count, cat) in enumerate(zip(df["Count"], df["Category"])):
        ax.text(
            count + max(1, count * 0.01),
            i,
            str(count),
            va="center",
            fontsize=7 if is_large else 10,
        )

    ax.set_title(f"Dataset Label Distribution ({CATEGORIES[0]}...)", pad=15)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("")

    if is_large:
        plt.yticks(fontsize=7)

    sns.despine(left=True, bottom=True)

    return _save_fig(fig, filename)


def generate_summary_table(zero_results, few_results):
    """Print a summary table to console."""
    k_values = set()
    for m in few_results:
        k_values.update(few_results[m].keys())
    k_values = sorted(k_values)

    header = f"{'Model':<20} {'Zero-Shot Acc':>14} {'Zero-Shot F1':>13}"
    for k in k_values:
        header += f" {f'{k}-shot Acc':>12}"

    lines = ["=" * 80, f"{'SUMMARY TABLE':^80}", "=" * 80, header, "─" * 80]

    for model in zero_results:
        acc = zero_results[model]["accuracy"]
        f1 = zero_results[model]["f1_weighted"]
        row = f"{model:<20} {acc:>14.4f} {f1:>13.4f}"
        for k in k_values:
            if model in few_results and k in few_results[model]:
                fs_acc = few_results[model][k]["accuracy_mean"]
                row += f" {fs_acc:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        lines.append(row)

    lines.append("─" * 80)
    logger.info("\n" + "\n".join(lines))


def plot_wise_ft(wise_ft_results, filename="wise_ft_curve.png"):
    """Line plot: accuracy vs mixing coefficient (alpha) for WiSE-FT."""
    records = []
    for model_name, alpha_results in wise_ft_results.items():
        for alpha, res in alpha_results.items():
            records.append(
                {
                    "Alpha": alpha,
                    "Accuracy": res["accuracy"],
                    "Model": model_name,
                }
            )

    if not records:
        return None

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Alpha",
        y="Accuracy",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette="deep",
        ax=ax,
        linewidth=2.5,
        markersize=10,
    )

    ax.set_ylim(0, min(1.05, df["Accuracy"].max() + 0.08))
    ax.set_title("WiSE-FT: Accuracy vs Mixing Coefficient (α)", pad=15)
    ax.set_xlabel("Mixing Coefficient α (0=Zero-Shot, 1=Fine-Tuned)")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Model")
    sns.despine(left=True)

    return _save_fig(fig, filename)
