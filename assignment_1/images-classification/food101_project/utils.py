from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

PathLike = Union[str, Path]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_json(data: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_markdown(text: str, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def save_history_plot(history: List[Dict[str, Any]], out_path: PathLike, title_prefix: str) -> None:
    if not history:
        return
    epochs = [x['epoch'] for x in history]
    train_loss = [x['train_loss'] for x in history]
    val_loss = [x['val_loss'] for x in history]
    train_acc = [x['train_acc'] for x in history]
    val_acc = [x['val_acc'] for x in history]
    train_f1 = [x['train_f1'] for x in history]
    val_f1 = [x['val_f1'] for x in history]

    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, train_loss, label='train_loss')
    ax1.plot(epochs, val_loss, label='val_loss')
    ax1.set_title(f'{title_prefix} Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, train_acc, label='train_acc')
    ax2.plot(epochs, val_acc, label='val_acc')
    ax2.set_title(f'{title_prefix} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, train_f1, label='train_f1')
    ax3.plot(epochs, val_f1, label='val_f1')
    ax3.set_title(f'{title_prefix} Macro-F1')
    ax3.set_xlabel('Epoch')
    ax3.legend()

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix_figure(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: PathLike, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    fig.colorbar(im)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def top_confusions(y_true: List[int], y_pred: List[int], class_names: List[str], top_k: int = 15) -> List[Tuple[str, str, int]]:
    cm = confusion_matrix(y_true, y_pred)
    pairs: List[Tuple[str, str, int]] = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def write_top_confusions_csv(rows: List[Tuple[str, str, int]], out_path: PathLike) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=['true_class', 'predicted_class', 'count'])
    df.to_csv(out_path, index=False)


def timestamp() -> str:
    return time.strftime('%Y%m%d-%H%M%S')


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0, 1)
    return image
