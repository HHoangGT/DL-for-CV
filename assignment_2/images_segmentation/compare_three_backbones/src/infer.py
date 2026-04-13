from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.misc import get_device
from src.utils.visualization import overlay_mask_on_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["system"].get("device", "cuda"))
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

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
    model.eval()

    image = Image.open(args.image).convert("RGB")
    resized = image.resize((ds_cfg["image_size"], ds_cfg["image_size"]), Image.BILINEAR)
    x = TF.to_tensor(resized)
    x = TF.normalize(x, ds_cfg["mean"], ds_cfg["std"]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()

    overlay = overlay_mask_on_image(resized, pred)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
