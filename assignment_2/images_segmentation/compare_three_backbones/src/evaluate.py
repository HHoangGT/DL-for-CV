from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.voc import VOCSegmentationDataset
from src.losses import build_loss
from src.models.deeplabv3plus import DeepLabV3Plus
from src.train import validate
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.misc import ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["system"].get("device", "cuda"))

    ds_cfg = cfg["dataset"]
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
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["system"]["num_workers"])

    model_cfg = cfg["model" ]
    model = DeepLabV3Plus(
        backbone_name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        low_level_index=model_cfg.get("low_level_index", 0),
        high_level_index=model_cfg.get("high_level_index", -1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = build_loss(ignore_index=cfg["dataset"]["ignore_index"])
    out_dir = ensure_dir(Path(args.checkpoint).resolve().parent / "evaluation")
    metrics = validate(model, val_loader, criterion, device, cfg, vis_dir=out_dir)
    print(metrics)


if __name__ == "__main__":
    main()
