from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from engine import evaluate
from utils import save_confusion_matrix_figure, save_json, top_confusions, write_top_confusions_csv

PathLike = Union[str, Path]


@torch.no_grad()
def evaluate_checkpoint(model, checkpoint_path: PathLike, test_loader, device, class_names, report_dir: PathLike, experiment_name: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    confusions = top_confusions(y_true, y_pred, class_names)
    summary = {
        'experiment_name': experiment_name,
        'test_loss': metrics['loss'],
        'test_accuracy': float(accuracy_score(y_true, y_pred)),
        'test_macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'test_macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'test_macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'top_confusions': confusions,
    }
    save_json(summary, report_dir / f'{experiment_name}_summary.json')
    text_report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(report_dir / f'{experiment_name}_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)
    save_confusion_matrix_figure(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_path=report_dir / f'{experiment_name}_confusion_matrix.png',
        title=f'Confusion Matrix - {experiment_name}',
    )
    write_top_confusions_csv(confusions, report_dir / f'{experiment_name}_top_confusions.csv')
    return summary


def write_comparison_csv(rows, out_path: PathLike) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
