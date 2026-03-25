from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE
from models import build_model
from utils import IMAGENET_MEAN, IMAGENET_STD, get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def load_image(image_path: str):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    display = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return tensor, display


def get_target_layer(model, model_name: str):
    if model_name == 'resnet50':
        return model.layer4[-1].conv3
    if model_name == 'efficientnet_b0':
        return model.features[-1][0]
    raise ValueError('Grad-CAM hiện chỉ hỗ trợ resnet50 và efficientnet_b0. Với ViT, bạn có thể dùng attention visualization như phần mở rộng khác.')


def overlay_heatmap(image: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    heatmap = cm.jet(cam_map)[..., :3]
    overlay = 0.45 * heatmap + 0.55 * image
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for Food-101 CNN models')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet50', 'efficientnet_b0'])
    parser.add_argument('--num_classes', type=int, default=101)
    parser.add_argument('--class_idx', type=int, default=-1)
    parser.add_argument('--output_path', type=str, default='outputs/reports/gradcam_overlay.png')
    args = parser.parse_args()

    device = get_device()
    model = build_model(args.model_name, num_classes=args.num_classes, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    target_layer = get_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)

    x, display = load_image(args.image_path)
    x = x.to(device)
    class_idx = None if args.class_idx < 0 else args.class_idx
    cam_map, pred_idx = gradcam(x, class_idx)
    overlay = overlay_heatmap(display, cam_map)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(display)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title(f'Grad-CAM (class idx={pred_idx})')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved Grad-CAM figure to: {out_path}')


if __name__ == '__main__':
    main()
