"""
Training script for PASCAL VOC Semantic Segmentation.
Supports U-Net and DeepLabV3+ with TensorBoard logging and checkpointing.

Usage:
    python train.py --arch unet --backbone resnet50 --epochs 50
    python train.py --arch deeplabv3plus --backbone resnet50 --epochs 50
    python train.py --arch unet --copy-paste --epochs 50
"""

import os
import time
import argparse
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.pascal_voc import get_dataloaders, NUM_CLASSES, IGNORE_INDEX
from models.builder import build_model, get_model_info
from utils.metrics import compute_miou, compute_dice, compute_pixel_accuracy
from utils.loss import get_loss_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Segmentation Model on PASCAL VOC"
    )

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="unet",
        choices=["unet", "deeplabv3plus"],
        help="Segmentation architecture",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="Encoder backbone name"
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pretrained weights for encoder",
    )

    # Data
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for VOC dataset"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download dataset if not present"
    )
    parser.add_argument("--crop-size", type=int, default=512, help="Training crop size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Augmentation
    parser.add_argument(
        "--copy-paste", action="store_true", help="Enable Copy-Paste augmentation"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement). 0 to disable.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "dice", "combined"],
        help="Loss function",
    )

    # Output
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Auto-detect if not set.",
    )

    return parser.parse_args()


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, writer, global_step
):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    total_miou = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)  # [B, C, H, W]
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        # Metrics
        preds = outputs.argmax(dim=1)  # [B, H, W]
        miou, _ = compute_miou(
            preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
        )

        running_loss += loss.item()
        total_miou += miou
        num_batches += 1
        global_step += 1

        # Log to TensorBoard every 10 batches
        if (batch_idx + 1) % 10 == 0:
            writer.add_scalar("Train/Loss_step", loss.item(), global_step)
            writer.add_scalar("Train/mIoU_step", miou, global_step)
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} | mIoU: {miou:.4f}"
            )

    avg_loss = running_loss / max(num_batches, 1)
    avg_miou = total_miou / max(num_batches, 1)

    return avg_loss, avg_miou, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation."""
    model.eval()
    running_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        preds = outputs.argmax(dim=1)
        miou, _ = compute_miou(
            preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
        )
        dice, _ = compute_dice(
            preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
        )
        pixel_acc = compute_pixel_accuracy(preds, masks, ignore_index=IGNORE_INDEX)

        running_loss += loss.item()
        total_miou += miou
        total_dice += dice
        total_pixel_acc += pixel_acc
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    avg_miou = total_miou / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_pixel_acc = total_pixel_acc / max(num_batches, 1)

    return avg_loss, avg_miou, avg_dice, avg_pixel_acc


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        },
        path,
    )
    print(f"  >> Checkpoint saved: {path}")


def main():
    args = parse_args()

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Directories ──
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Run name ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cp_suffix = "_copypaste" if args.copy_paste else ""
    run_name = f"{args.arch}_{args.backbone}{cp_suffix}_{timestamp}"
    log_path = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_path)
    print(f"TensorBoard logs: {log_path}")
    print(f"  -> Run: tensorboard --logdir {args.log_dir}")

    # ── Data ──
    print("Loading PASCAL VOC 2012 dataset...")
    crop_size = (args.crop_size, args.crop_size)
    train_loader, val_loader = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        crop_size=crop_size,
        num_workers=args.num_workers,
        download=args.download,
        copy_paste=args.copy_paste,
    )
    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    # ── Model ──
    print(f"Building model: {args.arch} (backbone: {args.backbone})")
    model = build_model(
        architecture=args.arch,
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        encoder_weights=args.encoder_weights,
    )
    model_info = get_model_info(model)
    model = model.to(device)

    # Log model info
    writer.add_text("Model/Architecture", args.arch)
    writer.add_text("Model/Backbone", args.backbone)
    writer.add_text("Model/TotalParams", f"{model_info['total_params_M']}M")

    # ── Loss, Optimizer, Scheduler ──
    criterion = get_loss_fn(
        loss_type=args.loss, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training Loop ──
    best_miou = 0.0
    global_step = 0
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_dice": [],
        "val_pixel_acc": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_miou, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            global_step,
        )

        # Validate
        val_loss, val_miou, val_dice, val_pixel_acc = validate(
            model, val_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        writer.add_scalar("Train/Loss_epoch", train_loss, epoch)
        writer.add_scalar("Train/mIoU_epoch", train_miou, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/mIoU", val_miou, epoch)
        writer.add_scalar("Val/Dice", val_dice, epoch)
        writer.add_scalar("Val/PixelAcc", val_pixel_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_dice"].append(val_dice)
        history["val_pixel_acc"].append(val_pixel_acc)

        # Print summary
        print(
            f"\nEpoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s) | "
            f"LR: {current_lr:.2e}\n"
            f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}\n"
            f"  Val   Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val PixelAcc: {val_pixel_acc:.4f}"
        )

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            best_path = os.path.join(
                args.ckpt_dir, f"{args.arch}_{args.backbone}_voc_best.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {"val_miou": val_miou, "val_dice": val_dice},
                best_path,
            )
            print(f"  ★ New best mIoU: {best_miou:.4f}")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            periodic_path = os.path.join(
                args.ckpt_dir, f"{args.arch}_{args.backbone}_voc_epoch{epoch}.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {"val_miou": val_miou, "val_dice": val_dice},
                periodic_path,
            )

        # Early Stopping Check
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"\n[!] Early stopping triggered after {epochs_without_improvement} epochs without improvement."
            )
            break

    # ── Save final model ──
    final_path = os.path.join(
        args.ckpt_dir, f"{args.arch}_{args.backbone}_voc_final.pth"
    )
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        args.epochs,
        {"val_miou": val_miou, "val_dice": val_dice},
        final_path,
    )

    # ── Save training history ──
    history_path = os.path.join(args.ckpt_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")

    writer.close()

    # ── Final Summary ──
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"  Architecture: {args.arch} | Backbone: {args.backbone}")
    print(f"  Best Val mIoU: {best_miou:.4f}")
    print(
        f"  Best checkpoint: {args.ckpt_dir}/{args.arch}_{args.backbone}_voc_best.pth"
    )
    print(f"  TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
