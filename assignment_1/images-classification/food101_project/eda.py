from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import DATA_ROOT, IMG_SIZE, REPORT_DIR, SEED
from dataset import build_splits
from utils import save_json, save_markdown


def sample_image_grid(df: pd.DataFrame, out_path: Path, num_classes: int = 12, seed: int = SEED) -> None:
    rng = random.Random(seed)
    labels = sorted(df['label_name'].unique().tolist())
    picked = rng.sample(labels, min(num_classes, len(labels)))
    fig = plt.figure(figsize=(14, 10))
    for idx, label in enumerate(picked, start=1):
        row = df[df['label_name'] == label].sample(1, random_state=seed).iloc[0]
        img = Image.open(row['img_path']).convert('RGB')
        ax = fig.add_subplot(3, 4, idx)
        ax.imshow(img)
        ax.set_title(label.replace('_', ' '), fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def image_size_stats(df: pd.DataFrame, max_samples: int = 3000) -> dict:
    sample_df = df if len(df) <= max_samples else df.sample(max_samples, random_state=SEED)
    widths = []
    heights = []
    for img_path in sample_df['img_path']:
        with Image.open(img_path) as img:
            widths.append(img.width)
            heights.append(img.height)
    return {
        'num_checked_images': len(sample_df),
        'width_min': int(min(widths)),
        'width_max': int(max(widths)),
        'width_mean': round(sum(widths) / len(widths), 2),
        'height_min': int(min(heights)),
        'height_max': int(max(heights)),
        'height_mean': round(sum(heights) / len(heights), 2),
        'training_input_size_after_transform': IMG_SIZE,
    }


def write_eda_markdown(summary: dict, out_path: Path) -> None:
    md = f"""# Food-101 EDA

## 1. Bài toán
- Loại bài toán: **image classification**.
- Input: ảnh RGB món ăn.
- Output: nhãn thuộc **101 lớp**.
- Dataset phù hợp đề bài vì có số lớp lớn hơn 5 và số lượng mẫu huấn luyện rất lớn.

## 2. Quy mô dữ liệu
- Tổng số ảnh sau khi tách nội bộ: **{summary['total_images_after_internal_split']:,}**
- Số lớp: **{summary['num_classes']}**
- Train / Val / Test: **{summary['train_images']:,} / {summary['val_images']:,} / {summary['test_images']:,}**

## 3. Cân bằng lớp
- Train balanced: **{summary['is_train_balanced']}**
- Val balanced: **{summary['is_val_balanced']}**
- Test balanced: **{summary['is_test_balanced']}**
- Mỗi lớp ở train có **{summary['train_distribution_min']}** ảnh.
- Mỗi lớp ở val có **{summary['val_distribution_min']}** ảnh.
- Mỗi lớp ở test có **{summary['test_distribution_min']}** ảnh.

## 4. Kích thước ảnh
- Đã kiểm tra **{summary['image_size_stats']['num_checked_images']:,}** ảnh mẫu.
- Width min / max / mean: **{summary['image_size_stats']['width_min']} / {summary['image_size_stats']['width_max']} / {summary['image_size_stats']['width_mean']}**
- Height min / max / mean: **{summary['image_size_stats']['height_min']} / {summary['image_size_stats']['height_max']} / {summary['image_size_stats']['height_mean']}**
- Ảnh sẽ được resize/crop về **{summary['image_size_stats']['training_input_size_after_transform']}x{summary['image_size_stats']['training_input_size_after_transform']}** để huấn luyện.

## 5. Nhận xét
- Đây là dataset đủ lớn và đủ khó để so sánh **CNN vs ViT**.
- Dữ liệu cân bằng nên **accuracy** là metric chính hợp lý; vẫn nên báo cáo thêm **macro-F1, precision, recall** và **confusion matrix**.
- Có thể mở rộng bằng **freeze backbone vs fine-tune full**, **augmentation vs no augmentation**, **Grad-CAM**, **demo Gradio**.
"""
    save_markdown(md, out_path)


def main():
    parser = argparse.ArgumentParser(description='EDA for Food-101')
    parser.add_argument('--data_root', type=str, default=str(DATA_ROOT))
    parser.add_argument('--output_dir', type=str, default=str(REPORT_DIR / 'eda'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx, train_df, val_df, test_df = build_splits(data_root=args.data_root)
    full_df = pd.concat(
        [
            train_df.assign(split='train'),
            val_df.assign(split='val'),
            test_df.assign(split='test'),
        ],
        ignore_index=True,
    )

    train_dist = train_df['label_name'].value_counts().sort_index()
    val_dist = val_df['label_name'].value_counts().sort_index()
    test_dist = test_df['label_name'].value_counts().sort_index()
    size_stats = image_size_stats(full_df)

    summary = {
        'num_classes': len(classes),
        'total_images_after_internal_split': int(len(full_df)),
        'train_images': int(len(train_df)),
        'val_images': int(len(val_df)),
        'test_images': int(len(test_df)),
        'train_distribution_min': int(train_dist.min()),
        'train_distribution_max': int(train_dist.max()),
        'val_distribution_min': int(val_dist.min()),
        'val_distribution_max': int(val_dist.max()),
        'test_distribution_min': int(test_dist.min()),
        'test_distribution_max': int(test_dist.max()),
        'is_train_balanced': bool(train_dist.nunique() == 1),
        'is_val_balanced': bool(val_dist.nunique() == 1),
        'is_test_balanced': bool(test_dist.nunique() == 1),
        'example_classes': classes[:10],
        'image_size_stats': size_stats,
    }

    save_json(summary, output_dir / 'dataset_summary.json')
    train_dist.to_csv(output_dir / 'train_distribution.csv', header=['count'])
    val_dist.to_csv(output_dir / 'val_distribution.csv', header=['count'])
    test_dist.to_csv(output_dir / 'test_distribution.csv', header=['count'])

    plt.figure(figsize=(20, 5))
    train_dist.plot(kind='bar')
    plt.title('Food-101 Train Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.tight_layout()
    plt.savefig(output_dir / 'train_label_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(20, 5))
    test_dist.plot(kind='bar')
    plt.title('Food-101 Test Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_label_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist([size_stats['width_mean']] * 5 + [size_stats['height_mean']] * 5)
    plt.title('Mean width/height proxy overview')
    plt.tight_layout()
    plt.savefig(output_dir / 'image_size_overview.png', dpi=200)
    plt.close()

    sample_image_grid(full_df, output_dir / 'sample_images_grid.png')
    write_eda_markdown(summary, output_dir / 'eda_report.md')

    print('EDA summary:')
    for k, v in summary.items():
        print(f'- {k}: {v}')
    print(f'EDA files saved to: {output_dir}')


if __name__ == '__main__':
    main()
