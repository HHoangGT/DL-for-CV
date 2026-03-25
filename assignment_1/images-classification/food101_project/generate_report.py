from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import DOCS_DIR, FIGURE_DIR, REPORT_DIR
from utils import save_markdown


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_report(run_name: str) -> str:
    report_dir = Path(REPORT_DIR) / run_name
    figure_dir = Path(FIGURE_DIR) / run_name
    comparison_csv = report_dir / 'comparison.csv'
    if not comparison_csv.exists():
        raise FileNotFoundError(f'Không tìm thấy {comparison_csv}')

    comparison_df = pd.read_csv(comparison_csv)
    rows_md = comparison_df.to_markdown(index=False)

    sections = [
        '# Assignment 1 - Image Classification (Food-101)',
        '',
        '## 1. Bài toán và dataset',
        '- Bài toán: phân loại ảnh món ăn Food-101.',
        '- Input: ảnh RGB.',
        '- Output: một trong 101 nhãn món ăn.',
        '- Mô hình so sánh: CNN vs ViT pretrained/fine-tune.',
        '',
        '## 2. Cấu hình thí nghiệm',
        f'- Run name: **{run_name}**',
        f'- Report dir: `{report_dir}`',
        f'- Figure dir: `{figure_dir}`',
        '',
        '## 3. Bảng so sánh kết quả',
        rows_md,
        '',
        '## 4. Nội dung cần đưa vào GitHub Pages / slide',
        '- EDA và mô tả dataset.',
        '- Dataset / Dataloader / Augmentation.',
        '- Xây dựng, huấn luyện, đánh giá, so sánh mô hình.',
        '- Kết quả thực nghiệm: bảng số liệu, biểu đồ, phân tích lỗi.',
        '- Mở rộng: freeze backbone, no augmentation, Grad-CAM, demo Gradio.',
        '',
        '## 5. Hình cần chèn',
        f'- `outputs/figures/{run_name}/*_history.png`',
        f'- `outputs/reports/{run_name}/*_confusion_matrix.png`',
        f'- `outputs/reports/{run_name}/*_top_confusions.csv`',
        '',
        '## 6. Nhận xét mẫu',
        '- So sánh accuracy và macro-F1 giữa CNN và ViT.',
        '- So sánh thời gian huấn luyện và số tham số trainable.',
        '- Phân tích lớp dễ nhầm lẫn dựa trên top-confusions.',
        '- Nêu ảnh hưởng của freeze backbone và augmentation.',
    ]

    for model_name in comparison_df['model'].tolist():
        full_summary_path = report_dir / f'{model_name}_full_summary.json'
        if full_summary_path.exists():
            summary = load_json(full_summary_path)
            sections.extend(
                [
                    '',
                    f'### {model_name}',
                    f"- Test accuracy: **{summary['test_accuracy']:.4f}**",
                    f"- Test macro-F1: **{summary['test_macro_f1']:.4f}**",
                    f"- Best val accuracy: **{summary['best_val_acc']:.4f}**",
                    f"- Best epoch: **{summary['best_epoch']}**",
                    f"- Training time: **{summary['training_time_hms']}**",
                ]
            )

    return '\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Generate markdown report from a training run')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='')
    args = parser.parse_args()

    markdown = build_report(args.run_name)
    output_path = Path(args.output_path) if args.output_path else DOCS_DIR / 'assignment1_image' / f'{args.run_name}_report.md'
    save_markdown(markdown, output_path)
    print(f'Saved report to: {output_path}')


if __name__ == '__main__':
    main()
