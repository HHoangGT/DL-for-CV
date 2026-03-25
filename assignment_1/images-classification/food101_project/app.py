from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE
from dataset import build_splits
from models import build_model
from utils import IMAGENET_MEAN, IMAGENET_STD, get_device


def load_inference_bundle(model_name: str, checkpoint: str, data_root: str):
    classes, _, _, _, _ = build_splits(data_root=data_root)
    device = get_device()
    model = build_model(model_name, num_classes=len(classes), pretrained=False, freeze_backbone=False)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    return model, classes, device, transform


def make_predict_fn(model, classes, device, transform):
    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return {}
        image = image.convert('RGB')
        x = transform(image).unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
        top5 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]
        return {label: float(score) for label, score in top5}

    return predict


def main():
    parser = argparse.ArgumentParser(description='Gradio demo for Food-101 classifier')
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet50', 'efficientnet_b0', 'vit_b_16'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    model, classes, device, transform = load_inference_bundle(args.model_name, args.checkpoint, args.data_root)
    predict_fn = make_predict_fn(model, classes, device, transform)

    title = f'Food-101 Demo - {args.model_name}'
    description = 'Upload một ảnh món ăn để xem top-5 dự đoán của mô hình.'
    app = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type='pil', label='Food image'),
        outputs=gr.Label(num_top_classes=5, label='Predictions'),
        title=title,
        description=description,
    )
    app.launch(share=args.share)


if __name__ == '__main__':
    main()
