from __future__ import annotations

import argparse

import gradio as gr
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
    parser.add_argument("--share", action="store_true")
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

    def predict(image: Image.Image):
        resized = image.convert("RGB").resize((ds_cfg["image_size"], ds_cfg["image_size"]), Image.BILINEAR)
        x = TF.to_tensor(resized)
        x = TF.normalize(x, ds_cfg["mean"], ds_cfg["std"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()
        return overlay_mask_on_image(resized, pred)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Input image"),
        outputs=gr.Image(type="pil", label="Segmentation overlay"),
        title="DeepLabV3+ Pascal VOC Demo",
        description=f"Backbone: {model_cfg['backbone']}",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
