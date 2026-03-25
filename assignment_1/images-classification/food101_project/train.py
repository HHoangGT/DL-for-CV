from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    FIGURE_DIR,
    REPORT_DIR,
    SEED,
    SUPPORTED_MODELS,
)
from dataset import create_dataloaders
from engine import train_model
from evaluate import evaluate_checkpoint, write_comparison_csv
from models import build_model, count_total_params, count_trainable_params
from utils import format_seconds, get_device, save_history_plot, save_json, set_seed, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN vs ViT on Food-101')
    parser.add_argument('--data_root', type=str, default=str(DATA_ROOT))
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'vit_b_16'])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()

    unknown = sorted(set(m for m in args.models if m not in SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f'Model không hỗ trợ: {unknown}. Các model hợp lệ: {SUPPORTED_MODELS}')
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run_name = args.run_name.strip() or timestamp()

    print(f'Using device: {device}')
    print(f'Run name: {run_name}')
    print(f'Freeze backbone: {args.freeze_backbone}')
    print(f'Use augmentation: {not args.no_augmentation}')

    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        seed=args.seed,
        use_augmentation=not args.no_augmentation,
    )
    class_names = metadata['classes']
    num_classes = metadata['num_classes']

    run_report_dir = Path(REPORT_DIR) / run_name
    run_figure_dir = Path(FIGURE_DIR) / run_name
    run_ckpt_dir = Path(CHECKPOINT_DIR) / run_name
    run_report_dir.mkdir(parents=True, exist_ok=True)
    run_figure_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            'run_name': run_name,
            'device': str(device),
            'models': args.models,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'num_workers': args.num_workers,
            'label_smoothing': args.label_smoothing,
            'freeze_backbone': args.freeze_backbone,
            'use_augmentation': not args.no_augmentation,
            'seed': args.seed,
            'num_classes': num_classes,
        },
        run_report_dir / 'run_config.json',
    )

    comparison_rows = []

    for model_name in args.models:
        print('\n' + '=' * 80)
        print(f'Training model: {model_name}')
        print('=' * 80)

        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=args.freeze_backbone,
        )
        model.to(device)

        ckpt_path = run_ckpt_dir / f'{model_name}_best.pth'
        train_out = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            checkpoint_path=ckpt_path,
            early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        )

        save_json(train_out, run_report_dir / f'{model_name}_train_history.json')
        save_history_plot(train_out['history'], run_figure_dir / f'{model_name}_history.png', model_name)

        best_model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
        summary = evaluate_checkpoint(
            model=best_model,
            checkpoint_path=ckpt_path,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            report_dir=run_report_dir,
            experiment_name=model_name,
        )

        total_params = count_total_params(model)
        trainable_params = count_trainable_params(model)

        summary['training_time_seconds'] = train_out['training_time_seconds']
        summary['training_time_hms'] = format_seconds(train_out['training_time_seconds'])
        summary['best_val_acc'] = train_out['best_val_acc']
        summary['best_epoch'] = train_out['best_epoch']
        summary['total_params'] = int(total_params)
        summary['trainable_params'] = int(trainable_params)
        save_json(summary, run_report_dir / f'{model_name}_full_summary.json')

        comparison_rows.append(
            {
                'run_name': run_name,
                'model': model_name,
                'freeze_backbone': args.freeze_backbone,
                'use_augmentation': not args.no_augmentation,
                'test_accuracy': summary['test_accuracy'],
                'test_macro_f1': summary['test_macro_f1'],
                'test_macro_precision': summary['test_macro_precision'],
                'test_macro_recall': summary['test_macro_recall'],
                'best_val_acc': train_out['best_val_acc'],
                'best_epoch': train_out['best_epoch'],
                'training_time_seconds': train_out['training_time_seconds'],
                'training_time_hms': format_seconds(train_out['training_time_seconds']),
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'checkpoint_path': str(ckpt_path),
            }
        )

    write_comparison_csv(comparison_rows, run_report_dir / 'comparison.csv')
    save_json({'rows': comparison_rows}, run_report_dir / 'comparison.json')

    print('\nDone.')
    print(f'Reports saved to: {run_report_dir}')
    print(f'Figures saved to: {run_figure_dir}')
    print(f'Checkpoints saved to: {run_ckpt_dir}')


if __name__ == '__main__':
    main()
