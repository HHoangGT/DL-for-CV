from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from src.datasets.voc import VOCSegmentationDataset
from src.losses import build_loss
from src.metrics import RunningSegmentationMetrics
from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.misc import ensure_dir, get_device, save_json, set_seed
from src.utils.visualization import save_prediction_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def segmentation_collate_fn(batch):
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        if key == "raw_image":
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = default_collate([item[key] for item in batch])

    return collated

def build_dataloaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    train_set = VOCSegmentationDataset(
        root_dir=ds_cfg["root_dir"],
        image_dir=ds_cfg["image_dir"],
        mask_dir=ds_cfg["mask_dir"],
        split_dir=ds_cfg["split_dir"],
        split_file=ds_cfg["train_split"],
        image_size=ds_cfg["image_size"],
        mean=ds_cfg["mean"],
        std=ds_cfg["std"],
        train=True,
    )
    val_set = VOCSegmentationDataset(
        root_dir=ds_cfg["root_dir"],
        image_dir=ds_cfg["image_dir"],
        mask_dir=ds_cfg["mask_dir"],
        split_dir=ds_cfg["split_dir"],
        split_file=ds_cfg["val_split"],
        image_size=ds_cfg["image_size"],
        mean=ds_cfg["mean"],
        std=ds_cfg["std"],
        train=False,
    )

    workers = cfg["system"]["num_workers"]
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
    )
    return train_loader, val_loader


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model"]
    model = DeepLabV3Plus(
        backbone_name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("backbone_pretrained", True),
        low_level_index=model_cfg.get("low_level_index", 0),
        high_level_index=model_cfg.get("high_level_index", -1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        img_size=cfg["dataset"]["image_size"],
    )
    return model.to(device)


def build_optimizer(cfg: dict, model: torch.nn.Module):
    train_cfg = cfg["train"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    name = train_cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, cfg):
    model.train()
    total_loss = 0.0
    use_amp = cfg["system"].get("amp", True) and device.type == "cuda"
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm", None)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, vis_dir: Path | None = None):
    model.eval()
    total_loss = 0.0
    metrics = RunningSegmentationMetrics(num_classes=cfg["model"]["num_classes"], ignore_index=cfg["dataset"]["ignore_index"])
    saved = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu()
        metrics.update(preds, masks.cpu())

        if vis_dir is not None and saved < cfg["evaluation"].get("num_visualizations", 8):
            for i in range(images.size(0)):
                if saved >= cfg["evaluation"].get("num_visualizations", 8):
                    break
                save_prediction_triplet(
                    batch["raw_image"][i],
                    masks[i].cpu().numpy(),
                    preds[i].numpy(),
                    vis_dir / f"{batch['id'][i]}.png",
                )
                saved += 1

    metric_obj = metrics.compute()
    return {
        "val_loss": total_loss / len(loader.dataset),
        "miou": metric_obj.miou,
        "dice": metric_obj.dice,
        "pixel_acc": metric_obj.pixel_acc,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = get_device(cfg["system"].get("device", "cuda"))
    exp_dir = ensure_dir(Path(cfg["output_dir"]) / cfg["experiment_name"])
    vis_dir = ensure_dir(exp_dir / "predictions")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg, device)
    criterion = build_loss(ignore_index=cfg["dataset"]["ignore_index"])
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler(enabled=cfg["system"].get("amp", True) and device.type == "cuda")

    history = []
    best_score = -1.0
    metrics_csv = exp_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "miou", "dice", "pixel_acc", "lr"])
        writer.writeheader()

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg)
            val_metrics = validate(model, val_loader, criterion, device, cfg, vis_dir if epoch == cfg["train"]["epochs"] else None)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "miou": val_metrics["miou"],
                "dice": val_metrics["dice"],
                "pixel_acc": val_metrics["pixel_acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history.append(row)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                    "metrics": row,
                },
                exp_dir / "last.pt",
            )

            if row[cfg["train"].get("save_best_by", "miou")] > best_score:
                best_score = row[cfg["train"].get("save_best_by", "miou")]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": cfg,
                        "metrics": row,
                    },
                    exp_dir / "best.pt",
                )

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_loss={row['val_loss']:.4f} miou={row['miou']:.4f} "
                f"dice={row['dice']:.4f} pixel_acc={row['pixel_acc']:.4f}"
            )

    save_json(history, exp_dir / "history.json")
    print(f"Finished. Artifacts saved to: {exp_dir}")


if __name__ == "__main__":
    main()
