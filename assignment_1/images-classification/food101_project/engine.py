from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

PathLike = Union[str, Path]


def _run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train() if train else model.eval()
    losses: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(loader, leave=False):
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()) * images.size(0))
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = float(np.sum(losses) / len(loader.dataset))
    epoch_acc = float(accuracy_score(all_labels, all_preds))
    epoch_f1 = float(f1_score(all_labels, all_preds, average='macro'))
    epoch_precision, epoch_recall, _, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0,
    )
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'macro_f1': epoch_f1,
        'macro_precision': float(epoch_precision),
        'macro_recall': float(epoch_recall),
        'y_true': all_labels,
        'y_pred': all_preds,
    }


def evaluate(model, loader, criterion, device):
    return _run_epoch(model, loader, criterion, optimizer=None, device=device, train=False)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    checkpoint_path: PathLike,
    early_stopping_patience: int = 3,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1
    wait = 0
    history: List[Dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = _run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)
        scheduler.step(val_metrics['accuracy'])
        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_f1': train_metrics['macro_f1'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['macro_f1'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        print(
            'Epoch {:02d}/{} | train_loss={:.4f} train_acc={:.4f} train_f1={:.4f} | '
            'val_loss={:.4f} val_acc={:.4f} val_f1={:.4f} | lr={:.2e}'.format(
                epoch,
                epochs,
                row['train_loss'],
                row['train_acc'],
                row['train_f1'],
                row['val_loss'],
                row['val_acc'],
                row['val_f1'],
                row['lr'],
            )
        )
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history,
                },
                checkpoint_path,
            )
            print(f'Saved best checkpoint -> {checkpoint_path}')
        else:
            wait += 1
            if wait >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    total_time = time.time() - started
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'training_time_seconds': total_time,
        'checkpoint_path': str(checkpoint_path),
    }
