import time

import torch
from torch import nn

from src.utils import AverageMeter, accuracy


def train_one_epoch(
    model: nn.Module, loader, optimizer, criterion, device: torch.device
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, criterion, device: torch.device
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: torch.device,
    epochs: int,
):
    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} | "
            f"time={row['epoch_time_sec']:.1f}s"
        )

    return history, best_state, best_val_acc
