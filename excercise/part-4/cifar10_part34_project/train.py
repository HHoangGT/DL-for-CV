import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from src.data import create_dataloaders
from src.engine import fit
from src.models.cnn_transformer import CNNTransformer
from src.models.vit_overlap import ViTOverlap
from src.models.vit_patch import ViTPatch
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_config,
    save_json,
    set_seed,
)


MODEL_REGISTRY = {
    "vit_patch": ViTPatch,
    "vit_overlap": ViTOverlap,
    "cnn_transformer": CNNTransformer,
}


def build_model(config):
    model_name = config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if model_name == "vit_patch":
        return ViTPatch(
            num_classes=config["num_classes"],
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
        )

    if model_name == "vit_overlap":
        return ViTOverlap(
            num_classes=config["num_classes"],
            image_size=config["image_size"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            padding=config["padding"],
        )

    return CNNTransformer(
        num_classes=config["num_classes"],
        image_size=config["image_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        dropout=config["dropout"],
    )


def save_history_csv(history, csv_path: Path):
    fieldnames = (
        list(history[0].keys())
        if history
        else [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_time_sec",
        ]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "cuda"))
    run_dir = ensure_dir(Path("results") / config["run_name"])

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 2),
        val_split=config.get("val_split", 0.1),
        seed=config.get("seed", 42),
    )

    # test_loader giữ lại để sau này predict Kaggle submission nếu cần
    _ = test_loader

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    history, best_state, best_val_acc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["epochs"],
    )

    best_ckpt_path = run_dir / "best.pt"
    torch.save(
        {
            "config": config,
            "state_dict": best_state,
            "best_val_acc": best_val_acc,
        },
        best_ckpt_path,
    )

    model.load_state_dict(best_state)

    best_epoch = max(history, key=lambda x: x["val_acc"]) if history else None

    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_epoch["val_loss"])
        if best_epoch is not None
        else None,
        "best_train_acc_at_best_val": float(best_epoch["train_acc"])
        if best_epoch is not None
        else None,
        "best_train_loss_at_best_val": float(best_epoch["train_loss"])
        if best_epoch is not None
        else None,
        "best_epoch": int(best_epoch["epoch"]) if best_epoch is not None else None,
        "num_parameters": count_parameters(model),
        "device": str(device),
        "note": "Kaggle test split has no ground-truth labels, so test accuracy/loss was not computed.",
    }

    save_history_csv(history, run_dir / "history.csv")
    save_json(metrics, run_dir / "metrics.json")
    save_json(config, run_dir / "used_config.json")

    print("Training finished.")
    print(metrics)
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print("Skipped test-set evaluation because Kaggle test split has no labels.")


if __name__ == "__main__":
    main()
