from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import DATA_ROOT, REPORT_DIR
from dataset import Food101CustomDataset, build_splits, get_eval_transform
from models import build_model
from utils import ensure_dir, get_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Visualize wrong predictions for Food-101')
    parser.add_argument('--data_root', type=str, default=str(DATA_ROOT))
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet50', 'efficientnet_b0', 'vit_b_16'])
    parser.add_argument('--num_samples', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default=str(REPORT_DIR / 'error_analysis'))
    args = parser.parse_args()

    classes, _, _, _, test_df = build_splits(data_root=args.data_root)
    ds = Food101CustomDataset(test_df, transform=get_eval_transform(), return_path=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_name, num_classes=len(classes), pretrained=False, freeze_backbone=False)
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    wrong = []
    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        for i in range(len(labels)):
            if preds[i].item() != labels[i].item():
                wrong.append(
                    {
                        'img_path': paths[i],
                        'true_idx': labels[i].item(),
                        'pred_idx': preds[i].item(),
                    }
                )
                if len(wrong) >= args.num_samples:
                    break
        if len(wrong) >= args.num_samples:
            break

    if not wrong:
        print('Không tìm thấy mẫu dự đoán sai trong số lượng đã duyệt.')
        return

    output_dir = ensure_dir(args.output_dir)
    fig = plt.figure(figsize=(14, 10))
    for i, item in enumerate(wrong, start=1):
        img = plt.imread(item['img_path'])
        ax = fig.add_subplot(3, 4, i)
        ax.imshow(img)
        ax.set_title(f"T: {classes[item['true_idx']]}\nP: {classes[item['pred_idx']]}")
        ax.axis('off')
    plt.tight_layout()
    out_path = Path(output_dir) / f'{args.model_name}_wrong_predictions.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved wrong prediction figure to: {out_path}')


if __name__ == '__main__':
    main()
