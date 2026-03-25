from __future__ import annotations

import argparse
import subprocess
import sys


EXPERIMENTS = {
    'baseline': [],
    'freeze_backbone': ['--freeze_backbone'],
    'no_augmentation': ['--no_augmentation'],
}


def main():
    parser = argparse.ArgumentParser(description='Run standard Food-101 experiments for the report')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'vit_b_16'])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='data')
    args = parser.parse_args()

    for name, extra_flags in EXPERIMENTS.items():
        cmd = [
            sys.executable,
            'train.py',
            '--run_name', name,
            '--data_root', args.data_root,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--num_workers', str(args.num_workers),
            '--models',
            *args.models,
            *extra_flags,
        ]
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f'Experiment failed: {name}')


if __name__ == '__main__':
    main()
