from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.voc import decode_segmap


def overlay_mask_on_image(image: Image.Image, pred_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    image = image.convert("RGB")
    color_mask = decode_segmap(pred_mask).convert("RGB")
    return Image.blend(image, color_mask, alpha=alpha)


def save_prediction_triplet(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    output_path: str | Path,
) -> None:
    image = image.convert("RGB")
    gt_img = decode_segmap(gt_mask).convert("RGB")
    pred_img = decode_segmap(pred_mask).convert("RGB")

    w, h = image.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(image, (0, 0))
    canvas.paste(gt_img, (w, 0))
    canvas.paste(pred_img, (w * 2, 0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
