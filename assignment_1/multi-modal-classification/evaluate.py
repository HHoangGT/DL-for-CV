"""
Evaluation utilities: accuracy, F1, confusion matrix, per-class metrics.
Results are saved as JSON for analysis.
"""

import json
import os

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from config import CATEGORIES, RESULTS_DIR


def compute_metrics(preds, labels, prefix=""):
    """
    Compute classification metrics.

    Args:
        preds: np.array of predicted labels
        labels: np.array of ground truth labels
        prefix: string prefix for result keys

    Returns:
        metrics: dict with accuracy, f1, precision, recall, confusion_matrix
    """
    # Overall metrics
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Per-class metrics
    unique_labels = sorted(set(labels) | set(preds))
    target_names = [CATEGORIES[i] for i in unique_labels if i < len(CATEGORIES)]
    report = classification_report(
        labels,
        preds,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    p = f"{prefix}_" if prefix else ""
    metrics = {
        f"{p}accuracy": float(acc),
        f"{p}f1_weighted": float(f1_weighted),
        f"{p}f1_macro": float(f1_macro),
        f"{p}precision_weighted": float(precision),
        f"{p}recall_weighted": float(recall),
        f"{p}confusion_matrix": cm.tolist(),
        f"{p}per_class": report,
        f"{p}num_samples": int(len(labels)),
    }

    return metrics


def print_metrics(metrics, title=""):
    """Pretty-print metrics to console using loguru."""
    # Find the key prefix
    keys = list(metrics.keys())
    prefix = ""
    for k in keys:
        if k.endswith("accuracy"):
            prefix = k.replace("accuracy", "")
            break

    lines = []
    if title:
        lines.append(f"{'─' * 50}")
        lines.append(f"  {title}")
        lines.append(f"{'─' * 50}")

    lines.append(f"  Accuracy:         {metrics[f'{prefix}accuracy']:.4f}")
    lines.append(f"  F1 (weighted):    {metrics[f'{prefix}f1_weighted']:.4f}")
    lines.append(f"  F1 (macro):       {metrics[f'{prefix}f1_macro']:.4f}")
    lines.append(f"  Precision (wt):   {metrics[f'{prefix}precision_weighted']:.4f}")
    lines.append(f"  Recall (wt):      {metrics[f'{prefix}recall_weighted']:.4f}")
    lines.append(f"  Samples:          {metrics[f'{prefix}num_samples']}")
    logger.info("\n".join(lines))


def save_results(results, filename):
    """Save results dict to JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            val = convert(obj)
            if val is not obj:
                return val
            return super().default(obj)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {filepath}")
    return filepath
